build_lib_example(
  NAME fncs-example
  SOURCE_FILES fncs-simulator.cc
  LIBRARIES_TO_LINK ${libcore}
                    ${libfncs}
                    ${libnetwork}
                    ${libinternet}
                    ${libpoint-to-point}
                    ${libapplications}
)

