/*
@file distance_sensor.h
@brief distance sensor driver definition for the Leebot robot
@author Adam Li
*/



namespace Leebot
{
	class DistanceSensorDriver
	{
	public:
		/*
		@brief Class constructor.
		@param distance: The maximum distance in cm that needs to be tracked
		*/
		DistanceSensorDriver(unsigned int distance) : maxDistance(distance) 
		{
		}
		
		/*
		@brief Return the distance to the nearest obstacle in cm
		@return The distance to the closest object in cm, or maxDistance if no object was detected
		*/
		virtual unsigned int getDistance() = 0;
	protected:
		unsigned int maxDistance;
	};
};