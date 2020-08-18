
#include "utils.h"

#include "globals.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <experimental/iterator>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <wx/wx.h>

// implementation
#include "vis_01_cMain.hpp"
cMain::cMain() : wxFrame(nullptr, wxID_ANY, "title") {}
cMain::~cMain() {}