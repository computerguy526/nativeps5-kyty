set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
set(KYTY_NPS5_SRC_SHA "unknown")
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		# 1: changes; another code is an error (a worktree whose submodule git dir does not
		# resolve), not a dirty tree.
		if(GIT_DIRTY_RESULT EQUAL 1)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
	endif()
endif()
# binaudit: tools-host/remote_build.sh builds a parentless snapshot commit (its hash says nothing about the source), so the real source commit
# comes in through NPS5_SRC_SHA (40 hex). Else a clean local checkout's own HEAD is the source commit. Embedded in the lib as 'NPS5_SRC_SHA=<sha>'
# (hybrid/kytyGpuBridge.cpp) and read by tools-host/buildid.py against components.lock.
if(DEFINED ENV{NPS5_SRC_SHA} AND "$ENV{NPS5_SRC_SHA}" MATCHES "^[0-9a-f]+$")
	set(KYTY_NPS5_SRC_SHA "$ENV{NPS5_SRC_SHA}")
elseif(NOT KYTY_GIT_REVISION STREQUAL "unknown")
	set(KYTY_NPS5_SRC_SHA "${KYTY_GIT_REVISION}")
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
