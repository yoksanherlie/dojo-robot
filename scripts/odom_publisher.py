#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
import math

class OdomPublisher():
    def __init__(self):
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=50)
        self.odom_broadcaster = tf.TransformBroadcaster()

    def run(self):
        r = rospy.Rate(1.0)
        current_time = rospy.Time.now()
        last_time = rospy.Time.now()
        x = 0.0
        y = 0.0
        th = 0.0

        vx = 0.1
        vy = -0.1
        vth = 0.1

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            dt = (current_time - last_time).to_sec()
            delta_x = (vx * math.cos(th) - vy * math.sin(th)) * dt
            delta_y = (vx * math.sin(th) + vy * math.cos(th)) * dt
            delta_th = vth * dt

            x += delta_x
            y += delta_y
            th += delta_h

            q = tf.transformations.quaternion_from_euler(0,0,th)
            odom_trans = TransformStamped()
            odom_trans.header.stamp = current_time
            odom_trans.header.frame_id = "odom"
            odom_trans.child_frame_id = "base_link"

            odom_trans.transform.translation.x = x
            odom_trans.transform.translation.y = y
            odom_trans.transform.translation.z = 0.0
            odom_trans.transform.rotation = q

            self.odom_broadcaster.sendTransform(odom_trans)

            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = "odom"

            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = q

            odom.child_frame_id = "base_link"
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = vth

            self.odom_pub.publish(odom)

            last_time = current_time
            r.sleep()

if __name__ == '__main__':
    rospy.init_node('odom_publisher', anonymous=True)

    odom_publisher = OdomPublisher()
    odom_publisher.run()

    #rospy.spin()
