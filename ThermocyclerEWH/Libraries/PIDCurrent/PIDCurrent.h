/*
  PIDCurrent.h - Library for changing the current of output PWM based on PID algorithm 
  and reaching a certain temperature
  Created by Adam Li, November 23rd, 2013.
  Released into public domain for EWH usage.
  */
  
  
#ifndef PIDCurrent_h
#define PIDCurrent_h

#include "Arduino.h"
#include "Heat_or_Cool.h"
#include "PID.h"
#include "LiquidCrystal.h"	//include LCD screen library

class PIDCurrent
{
  public:
    PIDCurrent(int cw, int ccw);
    void changeCurrent(int temp, long time);
    void reachTemp(int setPoint, long delayHold);


  private:
    int _cw;
    int _ccw;
    int _temp;
    long _time;
    int _setPoint;
    long _delayHold;
    int _goal_temp;
    int _out_owm;
};

#endif
