foreach (_required_variable IN ITEMS
        THOR_SOURCE_DIR
        THOR_DESTINATION_DIR
        THOR_GENERATED_HEADER
        THOR_VERSION_HEADER
        THOR_STAMP)
    if (NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} is required")
    endif ()
endforeach ()

file(REMOVE_RECURSE "${THOR_DESTINATION_DIR}")
file(MAKE_DIRECTORY "${THOR_DESTINATION_DIR}")

# Preserve the source-relative include hierarchy needed transitively by the
# explicitly exported DeepLearning/Api headers. Only headers are staged; source
# files and tests are deliberately excluded.
file(COPY "${THOR_SOURCE_DIR}/DeepLearning"
        DESTINATION "${THOR_DESTINATION_DIR}"
        FILES_MATCHING PATTERN "*.h")
file(COPY "${THOR_SOURCE_DIR}/Utilities"
        DESTINATION "${THOR_DESTINATION_DIR}"
        FILES_MATCHING PATTERN "*.h")
file(COPY "${THOR_GENERATED_HEADER}"
        DESTINATION "${THOR_DESTINATION_DIR}")
file(COPY "${THOR_VERSION_HEADER}"
        DESTINATION "${THOR_DESTINATION_DIR}")
file(TOUCH "${THOR_STAMP}")
