add_definitions(-DFNCS)
include_directories(/Users/davidt/FNCS_install/include)
set(fncs_source_files
    model/fncs-simulator-impl.cc
    model/fncs-application.cc
)

set(fncs_header_files
    model/fncs-simulator-impl.h
    model/fncs-application.h
)

set(FNCS_LIB_PATH /Users/davidt/FNCS_install/lib/libfncs.dylib)
set(ZMQ_LIB_PATH /opt/homebrew/lib/libzmq.dylib)
set(CZMQ_LIB_PATH /opt/homebrew/lib/libczmq.dylib)

build_lib(
  LIBNAME fncs
  SOURCE_FILES ${fncs_source_files}
  HEADER_FILES ${fncs_header_files}
  LIBRARIES_TO_LINK
    ${FNCS_LIB_PATH}
    ${ZMQ_LIB_PATH}
    ${CZMQ_LIB_PATH}
    ${libnetwork}
)
