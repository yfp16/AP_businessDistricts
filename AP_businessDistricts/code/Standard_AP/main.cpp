#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include "ap.h"
using namespace std;

// test
int main(int argc, char** argv)
{
  int prefType = 1;
  if (argc >= 2) {
    prefType = atoi(argv[1]);
  }
  set<int> out;
  vector<int> examplar = affinityPropagation(stdin, prefType);

  for (size_t i = 0; i < examplar.size(); ++i) {
      printf("%d ", examplar[i]);
      out.insert(examplar[i]);
        
  }


  printf("\n=> total size = %d\n", (int)out.size());

  puts("");
  return 0;
}
