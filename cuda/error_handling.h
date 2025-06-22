#ifndef ERROR_HANDLING_H
#define ERROR_HANDLING_H

#include <stdio.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

// When included this will be a globally available utility macro
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif
