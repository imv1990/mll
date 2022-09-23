configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLL_TEST_DEPENDS
mll
)

# Show progress bar!
list(APPEND LLVM_LIT_ARGS
    "-s"
)

add_lit_testsuite(check-mll "Running the MLL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLL_TEST_DEPENDS}
  )

set_target_properties(check-mll PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLL ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLL_TEST_DEPENDS}
)
