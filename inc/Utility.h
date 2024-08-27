#pragma once

#define miv(a, b) ((a) > (b) ? (b) : (a))
#define mav(a, b) ((a) < (b) ? (b) : (a))

#include <assert.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

typedef unsigned int ui;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define pb push_back
#define mp make_pair

const int INF = 1000000000;

//#define NDEBUG

/**** how to use: vector<string> v = split(s, "c")****/
vector<string> split(const string &s, const string &seperator) {
  vector<string> result;
  typedef string::size_type string_size;
  string_size i = 0;

  while (i != s.size()) {
    int flag = 0;
    while (i != s.size() && flag == 0) {
      flag = 1;
      for (string_size x = 0; x < seperator.size(); ++x)
        if (s[i] == seperator[x]) {
          ++i;
          flag = 0;
          break;
        }
    }
    flag = 0;
    string_size j = i;
    while (j != s.size() && flag == 0) {
      for (string_size x = 0; x < seperator.size(); ++x)
        if (s[j] == seperator[x]) {
          flag = 1;
          break;
        }
      if (flag == 0) ++j;
    }
    if (i != j) {
      result.push_back(s.substr(i, j - i));
      i = j;
    }
  }
  return result;
}

string integer_to_string(long long number) {
  std::vector<ui> sequence;
  if (number == 0) sequence.push_back(0);
  while (number > 0) {
    sequence.push_back(number % 1000);
    number /= 1000;
  }

  char buf[5];
  std::string res;
  for (unsigned int i = sequence.size(); i > 0; i--) {
    if (i == sequence.size())
      sprintf(buf, "%u", sequence[i - 1]);
    else
      sprintf(buf, ",%03u", sequence[i - 1]);
    res += std::string(buf);
  }
  return res;
}


