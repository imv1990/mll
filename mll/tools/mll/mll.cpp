#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h>
#include <mlir/Conversion/TosaToLinalg/TosaToLinalg.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/SparseTensor/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>

#include "mll/AST/AST.h"
#include "mll/Dialect/GPU/GPUDialect.h"
#include "mll/Dialect/OMP/OMPDialect.h"
#include "mll/Dialect/Vector/VectorDialect.h"
#include "mll/MLIRCodeGen/MLIRCodeGen.h"
#include "mll/Parse/BuiltinDialect.h"
#include "mll/Parse/Parser.h"

using namespace llvm;

cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::init("-"));

static cl::opt<bool> PrintIRAfterAll("pall", cl::desc("Print IR after all"),
                                     cl::init(false));

enum Action { EmitAST, EmitMLIR, EmitLLVM, EmitLLVMIR, Execute };

cl::opt<Action> ActionOpt(
    cl::desc("Choose IR Type:"),
    cl::values(clEnumValN(EmitAST, "emit-ast",
                          "Emit AST (after semantic checks)")),
    cl::values(clEnumValN(EmitMLIR, "emit-mlir", "Emit MLIR")),
    cl::values(clEnumValN(EmitLLVM, "emit-llvm", "Emit LLVM Dialect")),
    cl::values(clEnumValN(EmitLLVMIR, "emit-llvm-ir", "Emit LLVM IR")),
    cl::values(clEnumValN(Execute, "jit", "JIT the IR")), cl::init(Execute));

static int lowerToLLVMDialect(mlir::MLIRContext &context,
                              mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);

  if (PrintIRAfterAll) {
    context.disableMultithreading();
    pm.enableIRPrinting([](mlir::Pass *p, mlir::Operation *o) { return false; },
                        [](mlir::Pass *p, mlir::Operation *o) { return true; });
  }
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(mlir::arith::createArithmeticBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::tosa::createTosaToLinalg());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createLowerGpuOpsToNVVMOpsPass());
  pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  //   pm.addPass(mlir::createGpuToLLVMConversionPass());
  pm.addPass(mlir::createBufferizationToMemRefPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createMemRefToLLVMPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::arith::createConvertArithmeticToLLVMPass());

  mlir::LowerToLLVMOptions options(&context);
  pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

static int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerOpenMPDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

static int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerOpenMPDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  engineOptions.sharedLibPaths = {MLL_PRINT_LIB, MLIR_PRINT_LIB,
                                  "/usr/lib/llvm-10/lib/libomp.so",
                                  MLIR_C_PRINT_LIB};
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int main(int argc, char **argv) {

  InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure().failed();
  }

  mll::MLLContext context;

  mlir::MLIRContext *mlirContext = context.getMLIRContext();
  context.registerDialect<mll::builtin::BuiltinDialect>();
  context.registerDialect<mll::gpu::GPUDialect>();
  context.registerDialect<mll::omp::OMPDialect>();
  context.registerDialect<mll::vector::VectorDialect>();

  // Generates AST
  auto astModule = mll::parseInputFile(std::move(file), &context);
  if (!astModule) {
    llvm::errs() << "\n Failed compiling Input file! \n";
    return 1;
  }
  if (ActionOpt == EmitAST) {
    astModule->dump(llvm::outs());
    return 0;
  }

  SourceMgr sourceMgr;
  auto ownedBuffer = mlir::openInputFile(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                    context.getMLIRContext());

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (!mll::convertToMLIR(&context, module, astModule)) {
    llvm::errs() << "\n MLIR conversion failed \n";
    return 1;
  }

  auto result = mlir::verify(*module);
  if (result.failed()) {
    module->dump();
    llvm::errs() << "MLIR verification failed";
    return 1;
  }
  if (ActionOpt == EmitMLIR) {
    module->print(llvm::outs());
    return 0;
  }

  if (lowerToLLVMDialect(*mlirContext, module)) {

    module->dump();
    llvm::errs() << "\n LLVM dialect lowering failed";
    return 1;
  }

  if (ActionOpt == EmitLLVM) {
    module->print(llvm::outs());
    return 0;
  }

  if (ActionOpt == EmitLLVMIR) {
    return dumpLLVMIR(module.get());
  }
  assert(ActionOpt == Execute);

  return runJit(module.get());
}
