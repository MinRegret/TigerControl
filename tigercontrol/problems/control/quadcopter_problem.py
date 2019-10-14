# Problem class
# Author: John Hallman

from tigercontrol.problems.control import ControlProblem
import tigercontrol.problems.control.quadcopter as quadcopter
import tigercontrol.problems.control.quadcopter_controller as quadcopter_controller
import tigercontrol.problems.control.quadcopter_gui as quadcopter_gui
# import quadcopter, quadcopter_gui, quadcopter_controller
import signal
import sys
import argparse

# Constants
TIME_SCALING = 1.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds
run = True
GOALS = [(1,1,2),(1,-1,4),(-1,-1,2),(-1,1,4)]
YAWS = [0,3.14,-1.54,1.54]

class QuadcopterSingleP2PProblem(ControlProblem):
	''' Description: class for online control tests '''

	def __init__(self):
		self.initialized = False

	def initialize(self):
		self.initialized = True
		self.reset()
		return self.gui_object, self.quad_id, self.quad

	def step(self, motor_action):
		self.quad.set_motor_speeds(self.quad_id, motor_action)
		self.quad.update(QUAD_DYNAMICS_UPDATE)
		[x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.quad.get_state(self.quad_id)
		return [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot]

	def reset(self):
		# Set goals to go to
		self.goals = GOALS
		self.yaws = YAWS
		# Define the quadcopters
		self.QUADCOPTER={'q1':{'position':[1,0,4],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
		self.quad_id = 'q1'
		# Controller parameters
		self.CONTROLLER_PARAMETERS = {'Motor_limits':[4000,9000],
							'Tilt_limits':[-10,10],
							'Yaw_Control_Limits':[-900,900],
							'Z_XY_offset':500,
							'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
							'Linear_To_Angular_Scaler':[1,1,0],
							'Yaw_Rate_Scaler':0.18,
							'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
							}

		# Catch Ctrl+C to stop threads
		# signal.signal(signal.SIGINT, signal_handler)
		# Make objects for quadcopter, gui and controller
		self.quad = quadcopter.Quadcopter(self.QUADCOPTER)
		self.gui_object = quadcopter_gui.GUI(quads=self.QUADCOPTER)
		return
		# ctrl = quadcopter_controller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_PARAMETERS,quad_identifier='q1')
		# Start the threads
		# self.quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
		# ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
	'''
	def update_target(self,target):
		self.target = target

	def update_yaw_target(self,target):
		self.yaw_target = self.wrap_angle(target)

	def thread_run(self,update_rate,time_scaling):
		update_rate = update_rate*time_scaling
		last_update = self.get_time()
		while(self.run==True):
			time.sleep(0)
			self.time = self.get_time()
			if (self.time - last_update).total_seconds() > update_rate:
				self.step()
				last_update = self.time

	def start_thread(self,update_rate=0.005,time_scaling=1):
		self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
		self.thread_object.start()

	def stop_thread(self):
		self.run = False
	'''
def main():
	quad_problem = QuadcopterSingleP2PProblem()
	gui_object, quad_id, quad = quad_problem.initialize()
	for i in range(300):
		state = quad_problem.step([8000,0,8000,0])
		print("state:" + str(state))
		gui_object.quads[quad_id]['position'] = quad.get_position(quad_id)
		gui_object.quads[quad_id]['orientation'] = quad.get_orientation(quad_id)
		gui_object.update()

if __name__ == "__main__":
	main()
	
