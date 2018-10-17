/*
@file newping_distance_sensor.h
@brief distance sensor driver for the distance sensors supported by NewPing library, such as the
HC-SR04
@author Adam Li

*/

#include <distance_sensor.h>

namespace Leebot
{
	class DistanceSensor : public DistanceSensorDriver
	{
	public:
		DistanceSensor(int triggerPin, int echoPin, int maxDistance)
			: DistanceSensorDriver(maxDistance), 
			sensor(triggerPin, echoPin, maxDistance)
		{
		}
		
		virtual unsigned int getDistance()
		{
			unsigned int distance = sensor.ping_cm();
			if(distance <= 0)
				return maxDistance;
			return distance;
		}
	private:
		NewPing sensor;
	};
};