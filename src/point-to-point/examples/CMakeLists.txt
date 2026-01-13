build_lib_example(
  NAME main-attribute-value
  SOURCE_FILES main-attribute-value.cc
  LIBRARIES_TO_LINK ${libcore}
                    ${libfncs}
                    ${libnetwork}
                    ${libinternet}
                    ${libpoint-to-point}
                    ${libcsma}
                    ${libbridge}
                    ${libapplications}
)
