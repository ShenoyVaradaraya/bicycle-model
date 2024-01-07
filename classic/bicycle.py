# Copyright @2024 Varadaraya Ganesh Shenoy
# Using Kinematic Bicycle Model to go from start and goal

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class Vehicle():
    """
    Vehicle specific parameters
    """

    def __init__(self, L):
        self.L = L
        # self.max_alpha = max_steering
        self.start = [10, 20, 0]  # [x,y,theta] (2D-pose)
        self.goal = [20, 5, 0]


class BicycleModel():
    """
    Implementation of Bicycle Model
    """

    def __init__(self, L):
        self.vehicle = Vehicle(L)
        self.reached = False

    def update(self, current_state, delta, dt):
        """
        Updating the pose of the vehicle 
        """
        x, y, theta = current_state

        new_x = x + dt*np.cos(theta)
        new_y = y + dt*np.sin(theta)
        new_theta = theta + dt*np.tan(delta)/self.vehicle.L * np.abs(dt)
        
        return [new_x,new_y,new_theta]

    def bicycle(self, dt):
        """
        Function that generates the change in poses according to bicycle model
        """
        current_state = self.vehicle.start
        path=[current_state]

        while not self.reached_goal(current_state, self.vehicle.goal):
            goal_heading = np.arctan2(
                self.vehicle.goal[1]-current_state[1], self.vehicle.goal[0]-current_state[0])

            delta = goal_heading-current_state[2]

            current_state = self.update(current_state, delta, dt)
            path.append(current_state)
        print(path)
        return path
        

    def reached_goal(self, current_state, goal, tolerance=0.1):
        return np.sqrt((current_state[0] - goal[0]) ** 2 + (current_state[1] - goal[1]) ** 2) < tolerance


if __name__ == "__main__":
    bm = BicycleModel(1)
    path = bm.bicycle(0.5)

    path = list(zip(*path))
    plt.plot(path[0], path[1], marker='o')
    plt.title('Bicycle Navigation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
