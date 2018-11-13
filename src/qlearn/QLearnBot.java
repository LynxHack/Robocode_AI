package qlearn;

import robocode.Robot;
import robocode.ScannedRobotEvent;

public class QLearnBot extends Robot{
    public void run() {
        turnRight(getHeading() % 90);
        turnGunRight(90);
        while(true) {
            ahead(1000);
            turnRight(90);
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        fire(1);
    }
}