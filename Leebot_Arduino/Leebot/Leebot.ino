/*
@file leebot.ino
@brief Arduino robot vehicle firmware
@author Adam Li

*/

#define ENABLE_ARDUINO_MOTOR_DRIVER
#define ENABLE_NEWPING_DISTANCE_SENSOR_DRIVER
#define ENABLE_BT_SERIAL
#define LOGGING

#define RUN_TIME 30
#define TOO_CLOSE 10
#define MAX_DISTANCE (TOO_CLOSE * 20)

#include "logging.h"

#ifdef ENABLE_BT_SERIAL
#include<SoftwareSerial.h>
#define BT_RX_PIN 16
#define BT_TX_PIN 17
SoftwareSerial BTSerial(BT_RX_PIN, BT_TX_PIN);
#endif

#ifdef ENABLE_ARDUINO_MOTOR_DRIVER
#include "arduino_motor_driver.h"
#define RIGHT_MOTOR_INIT 1 //1 == A, 2 == B
#define LEFT_MOTOR_INIT 2
#endif

#ifdef ENABLE_NEWPING_DISTANCE_SENSOR_DRIVER
#include <NewPing.h>
#include "newping_distance_sensor.h"
#define DISTANCE_SENSOR_INIT 7, 4, MAX_DISTANCE
#endif

#include "moving_average.h"

namespace Leebot
{
  class Robot
  {
  public:
    /*
    @brief Class constructor
     */
     Robot()
       : leftMotor(LEFT_MOTOR_INIT), rightMotor(RIGHT_MOTOR_INIT),
         distanceSensor(DISTANCE_SENSOR_INIT),
         distanceAverage(MAX_DISTANCE)
     {
       initialize();
     }
     
          
     /*
       @brief Initialize the robot states of LEFT/RIGHT motor
     */
     void initialize()
     {
       endTime = millis() + RUN_TIME * 1000;
       move();
     }
     
     /*
       @brief Update the state of the robot based on input from sensor and remote control.
       Must be called repeatedly while the robot is in operation.
     */
     void run()
     {
       if(stopped())
         return;
       
       unsigned long currentTime = millis();
       //unsigned long elapsedTime = currentTime - startTime;
       unsigned int distance = distanceAverage.add(distanceSensor.getDistance());
       log("state: %d, currentTime: %ul, distance: %u\n", state, currentTime, distance);
     
       if(doneRunning(currentTime))
         stop();
       else if(moving()) {
         if(obstacleAhead(distance))
           turn(currentTime);
       }
       else if(turning()) {
         if(doneTurning(currentTime, distance))
           move();
       }
     }
     
  protected:
     bool moving() { 
       return (state == stateMoving);
     }
     bool turning() {
       return (state == stateTurning);
     }
     bool stopped() {
       return (state == stateStopped);
     }
     void move()
     {
       leftMotor.setSpeed(255);
       rightMotor.setSpeed(255);
       state = stateMoving;
     }
     void stop()
     {
       leftMotor.setSpeed(0);
       rightMotor.setSpeed(0);
       state = stateStopped;
     }
     bool doneRunning(unsigned long currentTime)
     {
       return (currentTime >= endTime);
     }
     bool obstacleAhead(unsigned int distance)
     {
       return (distance <= TOO_CLOSE);
     }
     bool turn(unsigned long currentTime)
     {
       if(random(2) == 0) {
         leftMotor.setSpeed(-255);
         rightMotor.setSpeed(255);
       }
       else {
         leftMotor.setSpeed(255);
         rightMotor.setSpeed(-255);
       }
       state = stateTurning;
       endStateTime = currentTime + random(500, 1000);
     }
     bool doneTurning(unsigned long currentTime, unsigned int distance)
     {
       if (currentTime >= endStateTime)
         return (distance > TOO_CLOSE);
       return false;
     }
       
  private:
    Motor leftMotor;            //Motor initialization for leftMotor
    Motor rightMotor;           //Motor initialization for rightMotor
    DistanceSensor distanceSensor;
    MovingAverage<unsigned int, 3> distanceAverage;
   
    enum state_t { stateStopped, stateRunning, stateTurning, stateMoving };  //enumeration of robot state
    state_t state;              //initialization of that state
    unsigned long endTime;      //defines how long this will program will run
    unsigned long endStateTime; //indicate when action ends
  };
};

Leebot::Robot robot;

void setup()
{
  Serial.begin(9600);
  BTSerial.begin(9600);
  robot.initialize();
}

void loop()
{
  robot.run();
}
