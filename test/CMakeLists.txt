find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set( LIBS
  gtest
  gtest_main
  pthread
  )

set( progs
  BaggedDatasetTest
  )

foreach( prog ${progs} )
  add_executable( ${prog} ${prog}.cxx )
  target_link_libraries( ${prog} ${LIBS} )
  install( TARGETS ${prog} DESTINATION test )
  add_test( NAME ${prog} COMMAND ${prog} --gtest_repeat=1 )
endforeach( prog )
