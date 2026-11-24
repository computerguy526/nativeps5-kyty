# Test of src/generate_version.cmake (ctest generate_version_sha): NPS5_SRC_SHA overrides the commit recorded for the build
# (tools-host/remote_build.sh builds a parentless snapshot, so the checkout's own HEAD is not the source commit), a plain run in a
# git checkout falls back to its HEAD, and a non-hex value is ignored. Run: cmake -DSRC=<src dir> -DGIT=<git> -DREPO=<checkout> -P this
cmake_minimum_required(VERSION 3.20)
set(work "${CMAKE_CURRENT_BINARY_DIR}/gvs_test")
file(MAKE_DIRECTORY "${work}")
file(WRITE "${work}/in.h.in" "#define KYTY_NPS5_SRC_SHA \"@KYTY_NPS5_SRC_SHA@\"\n#define KYTY_GIT_REVISION \"@KYTY_GIT_REVISION@\"\n")

function(run_case name env_sha expect_regex)
	if(env_sha STREQUAL "")
		unset(ENV{NPS5_SRC_SHA})
	else()
		set(ENV{NPS5_SRC_SHA} "${env_sha}")
	endif()
	execute_process(COMMAND ${CMAKE_COMMAND} -D INPUT_FILE=${work}/in.h.in -D OUTPUT_FILE=${work}/${name}.h
	                -D GIT_EXECUTABLE=${GIT} -D GIT_WORKING_DIRECTORY=${REPO} -P ${SRC}/generate_version.cmake RESULT_VARIABLE rc)
	file(READ "${work}/${name}.h" out)
	if(NOT rc EQUAL 0 OR NOT out MATCHES "${expect_regex}")
		message(FATAL_ERROR "case ${name}: expected ${expect_regex}, got: ${out}")
	endif()
endfunction()

set(sha "0123456789abcdef0123456789abcdef01234567")
run_case(env_override "${sha}" "KYTY_NPS5_SRC_SHA \"${sha}\"")
run_case(env_garbage "not-a-sha; rm" "KYTY_NPS5_SRC_SHA \"[0-9a-f]+\"")
run_case(no_env "" "KYTY_NPS5_SRC_SHA \"[0-9a-f]+\"")
# the override must not leak into the full-commit identity used for compatibility checks
run_case(env_keeps_revision "${sha}" "KYTY_GIT_REVISION \"[0-9a-f]+\"")
message(STATUS "generate_version_sha: ok")
