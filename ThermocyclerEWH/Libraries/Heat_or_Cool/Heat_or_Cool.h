/*
  Heat_or_Cool.h - Library for either heating or cooling the AL block.
  Created by Adam Li, November 9th, 2013.
  Released into public domain for EWH usage.
  */
  
  
#ifndef Heat_or_Cool_h
#define Heat_or_Cool_h

#include "Arduino.h"

class Heat_or_Cool
{
  public:
    Heat_or_Cool(int cw, int ccw);
    void heat();
    void cool();
  private:
    int _cw;
    int _ccw;
};

#endif
