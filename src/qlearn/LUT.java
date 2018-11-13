package qlearn;

import java.awt.geom.Point2D;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import robocode.Robot;
import robocode.AdvancedRobot;
import robocode.BulletHitBulletEvent;
import robocode.BulletHitEvent;
import robocode.BulletMissedEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

public class LUT extends AdvancedRobot{
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