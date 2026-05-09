find_package(ArborX REQUIRED) # Try to find Kokkos externally

if(ArborX_FOUND)
	
	message(STATUS "Found ArborX: ${ArborX_DIR} (version \"${ArborX_VERSION}\")")

else()
	message(STATUS "ArborX not found externally. Please install ArborX first.")
endif()

target_link_libraries(libnukexc INTERFACE ArborX::ArborX)
