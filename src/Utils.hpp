#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cstdlib>
#include <signal.h>


static bool exit_app = false;

// Handle the CTRL-C keyboard signal

void nixExitHandler(int s)
{
    exit_app = true;
}

// Set the function to handle the CTRL-C
void setCtrlHandler()
{
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nixExitHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
}

#endif  // UTILS_HPP_