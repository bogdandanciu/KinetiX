#include <vector>
#include <string>
#include <float.h>
#include <cassert>
#include <unistd.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <fcntl.h>
#include <libgen.h>

#include "nrssys.hpp"
#include "platform.hpp"
#include "_nekCRF.hpp"
#include "nekCRF.hpp"

///////////////////////////////////////////////////////////////////////////////
//                                  API 
///////////////////////////////////////////////////////////////////////////////

void nekCRF::foo() 
{
  occa::memory q = platform->device.malloc(1);
  std::cout << "calling foo() ...\n";
}
