package qlearn;

import java.awt.geom.Point2D;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

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

//@xr add up 一个class有500多行，应该分成多个class的，这样看起来也更有条理

public class LUT extends AdvancedRobot implements LUTInterface {

    static File savetest = new File("./savetest.txt");
    static File convergeCheck = new File("./convergeCheck.txt");
    static File rewardCheck = new File("./rewardCheck.txt");
    static int battleCount = 1;

    public static double reward = 0;
    public static double totalReward = 0;
    // define state constant
    public static final int numHeading = 4;
    public static final int numEnemyBearing = 4;
    public static final int numEnemyDistance = 20;
    public static final int numMyEnergy = 10;
    public static final int numEnemyEnergy = 10;
    public static int stateIndex[][][][][] = new int[numHeading][numEnemyBearing][numEnemyDistance][numMyEnergy][numEnemyEnergy];

    private static final int numState = numHeading * numEnemyBearing
            * numEnemyDistance * numMyEnergy * numEnemyEnergy;
    private static final int numAction = 6;
    static double[][] qtable = new double[numState][numAction];

    // action parameters
    static int[] actionIndex = new int[6];

    // make actions also discrete, in accordance to discrete states
    // so that state-action pair is set up
    public static final int goAhead = 0;
    public static final int goBack = 1;
    public static final int goAheadTurnLeft = 2;
    public static final int goAheadTurnRight = 3;
    public static final int goBackTurnLeft = 4;
    public static final int goBackTurnRight = 5;
    public static final int numActions = 6;

    // pre-define the robot action parameters for all above actions
    public static final double RobotMoveDistance = 300.0;
    public static final double RobotTurnDegree = 45.0;

    // learning parameters
    private boolean first = true;
    public static final double LearningRate = 0.1;
    public static final double DiscountRate = 0.9;
    public static final double ExplorationRate = 0.6;
    public static final double ExploitationRate = 0.6;
    private int lastStateIndex;
    private int lastAction;

    // enemy parameters
    String nameEnemy;
    public double bearingEnemy;
    public double headingEnemy;
    public double spotTimeEnemy;
    public double speedEnemy;
    public double XPositionEnemy;
    public double YPositionEnemy;
    public double distanceEnemy = 10000;
    public double constantHeadingEnemy;
    public double energyEnemy;
    // fire system
    private double firePower = 1;
    private double direction = 500;

    public void run() {
        try {
            // to test loading and saving function
            load(savetest);
        } catch (IOException e) {
            e.printStackTrace();
        }
        initialiseLUT();
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);

        while (true) {
            move();
            firePower = 400 / distanceEnemy;
            radarMove();
            gunMove();
            if (getGunHeat() == 0) {
                setFire(firePower);
            }
            execute();
        }
    }

    private void gunMove() {
        long time;
        long nextTime;
        Point2D.Double p;
        p = new Point2D.Double(XPositionEnemy, YPositionEnemy);
        for (int i = 0; i < 20; i++) {
            nextTime = (int) Math
                    .round((getrange(getX(), getY(), p.x, p.y) / (20 - (3 * firePower))));
            time = getTime() + nextTime - 10;
            p = futurePosition(time);
        }
        // offsets the gun by the angle to the next shot based on linear
        // targeting provided by the enemy class
        double gunOffset = getGunHeadingRadians()
                - (Math.PI / 2 - Math.atan2(p.y - getY(), p.x - getX()));
        setTurnGunLeftRadians(nomalizeDegree(gunOffset));
    }

    public Point2D.Double futurePosition(long futureTime) {
        double duration = futureTime - spotTimeEnemy;
        double futureX;
        double futureY;
        if (Math.abs(constantHeadingEnemy) > 0.000001) {
            double radius = speedEnemy / constantHeadingEnemy;
            double tothead = duration * constantHeadingEnemy;
            futureY = YPositionEnemy
                    + (Math.sin(headingEnemy + tothead) * radius)
                    - (Math.sin(headingEnemy) * radius);
            futureX = XPositionEnemy + (Math.cos(headingEnemy) * radius)
                    - (Math.cos(headingEnemy + tothead) * radius);
        } else {
            futureY = YPositionEnemy + Math.cos(headingEnemy) * speedEnemy
                    * duration;
            futureX = XPositionEnemy + Math.sin(headingEnemy) * speedEnemy
                    * duration;
        }
        return new Point2D.Double(futureX, futureY);
    }

    private double getrange(double x1, double y1, double x2, double y2) {
        double xo = x2 - x1;
        double yo = y2 - y1;
        double h = Math.sqrt(xo * xo + yo * yo);
        return h;
    }

    private void radarMove() {
        double turnDegree;
        if (getTime() - spotTimeEnemy > 4) {
            turnDegree = Math.PI * 4;
        } else {
            turnDegree = getRadarHeadingRadians()
                    - (Math.PI / 2 - Math.atan2(YPositionEnemy - getY(),
                    XPositionEnemy - getX()));

            turnDegree = nomalizeDegree(turnDegree);
            if (turnDegree < 0)
                turnDegree -= Math.PI / 10;
            else
                turnDegree += Math.PI / 10;
        }

        setTurnRadarLeftRadians(turnDegree);
    }

    private double nomalizeDegree(double ang) {
        if (ang > Math.PI)
            ang -= 2 * Math.PI;
        if (ang < -Math.PI)
            ang += 2 * Math.PI;
        return ang;
    }

    private void move() {

        // normolize heading
        double tmpHeading = getHeading();
        tmpHeading += 45;
        if (tmpHeading > 360)
            tmpHeading -= 360;
        int heading = (int) (tmpHeading / 90);
        if (heading > numHeading - 1)
            heading = numHeading - 1;

        // normalize enemy bearing
        if (bearingEnemy < 0)
            bearingEnemy += 360;
        double tmpBearingEnemy = bearingEnemy + 45;
        if (tmpBearingEnemy > 360)
            tmpBearingEnemy -= 360;
        int enemyBearing = (int) (tmpBearingEnemy / 90);
        if (enemyBearing > numEnemyBearing - 1)
            enemyBearing = numEnemyBearing - 1;

        int enemyDistance = (int) (distanceEnemy / 30);
        // check array boundary
        if (enemyDistance > numEnemyDistance - 1)
            enemyDistance = numEnemyDistance - 1;

        int myEnegy = (int) (getEnergy() / 10);
        if (myEnegy > numMyEnergy - 1)
            myEnegy = numMyEnergy - 1;

        int enemyEnergy = (int) (energyEnemy / 10);
        if (enemyEnergy > numEnemyEnergy - 1)
            enemyEnergy = numEnemyEnergy - 1;

        double[] currentState = { heading, enemyBearing, enemyDistance,
                myEnegy, enemyEnergy };
        double crrentAction = train(currentState, reward);
        int intAction = (int) crrentAction;

        reward = 0;
        executeAction(intAction);
    }

    // RL for movement selection
    private void executeAction(int action) {
        switch (action) {
            case goAhead:
                setAhead(RobotMoveDistance);
                break;

            case goBack:
                setBack(RobotMoveDistance);
                break;

            case goAheadTurnLeft:
                setAhead(RobotMoveDistance);
                setTurnLeft(goAhead);
                break;

            case goAheadTurnRight:
                setAhead(RobotMoveDistance);
                setTurnRight(RobotTurnDegree);
                break;

            case goBackTurnLeft:
                setBack(RobotMoveDistance);
                setTurnRight(RobotTurnDegree);
                break;

            case goBackTurnRight:
                setBack(RobotMoveDistance);
                setTurnRight(RobotTurnDegree);
                break;
        }
    }


    public double outputFor(double[] X) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    // return the actionIndex to take for state X
    public double train(double[] X, double reward) {
        int stateIndexNow = indexFor(X);
        double maxQ = maxQValueFor(stateIndexNow);

        // select action policy
        int actionNow = selectAction(stateIndexNow);

        // ??? ��-greedy to choose action?
        // int action = actionMaxQValue(stateIndexNow);
        double diffQValue = 0;

        System.out.println("Reward: " + reward);
        if (first)
            first = false;
        else {
            // for sarsa learning
            // double sarsaQValue = qtable[stateIndexNow][actionNow];
            // double oldQValue = qtable[lastStateIndex][lastAction];
            // double newQValue = (1 - LearningRate) * oldQValue + LearningRate
            // * (reward + DiscountRate * sarsaQValue);

            // for Q-Learning
            double oldQValue = qtable[lastStateIndex][lastAction];
            double newQValue = (1 - LearningRate) * oldQValue + LearningRate
                    * (reward + DiscountRate * maxQ);
            diffQValue = newQValue - oldQValue;
            qtable[lastStateIndex][lastAction] = newQValue;
        }
        lastStateIndex = stateIndexNow;
        lastAction = actionNow;

        // measure learning and converging
        if (stateIndexNow == 1199 && actionNow == 0)
            saveDiffQValue(stateIndexNow, actionNow, diffQValue, convergeCheck);

        // ??? exploratory move here

        return actionNow;
    }

    private int selectAction(int stateIndexNow) {
        double qValue;
        double sum = 0.0;
        double[] value = new double[numActions];
        for (int i = 0; i < numActions; i++) {
            qValue = qtable[stateIndexNow][i];
            value[i] = Math.exp(ExploitationRate * qValue);
            sum += value[i];
        }

        if (sum != 0)
            for (int i = 0; i < value.length; i++) {
                value[i] /= sum;
            }
        else
            return actionMaxQValue(stateIndexNow);

        int action = 0;
        double cumProb = 0.0;
        double randomNum = Math.random();
        while (randomNum > cumProb && action < value.length) {
            cumProb += value[action];
            action++;
        }
        return action - 1;
    }

    private void saveDiffQValue(int stateIndexNow, double action,
                                double diffQValue, File convergeCheck) {
        BufferedWriter writer = null;
        try {
            // ture means append, not clear all
            writer = new BufferedWriter(new FileWriter(convergeCheck, true));
            writer.write(battleCount + "\t" + stateIndexNow + "\t" + action
                    + "\t" + diffQValue + "\n");

        } catch (IOException e) {
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (IOException e) {
            }
        }
    }

    @Override
    public void save(File argFile) {
        BufferedWriter writer = null;
        try {
            // ***ture means append, not clear all
            // writer = new BufferedWriter(new FileWriter(argFile, true));

            // ***not append, clear all and then write
            writer = new BufferedWriter(new FileWriter(argFile));
            for (int i = 0; i < numState; i++)
                for (int j = 0; j < numAction; j++) {
                    // writer.write("state: " + i + "\t" + "action: " + j + "\t"
                    // + "qvalue: " + qtable[i][j] + "\n");
                    // to be pure double numbers for easy loading
                    writer.write(i + "\t" + "\t" + j + "\t" + qtable[i][j]
                            + "\n");
                }

        } catch (IOException e) {
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (IOException e) {
            }
        }

    }


    public void load(File inputFile) throws IOException {
        try {
            FileInputStream fileInputStream = new FileInputStream(inputFile);
            Scanner inputScanner = new Scanner(fileInputStream);

            for (int i = 0; i < numState; i++)
                for (int j = 0; j < numAction; j++) {
                    double statee = inputScanner.nextDouble();
                    double actionn = inputScanner.nextDouble();
                    if ((int) statee == i && (int) actionn == j)
                        qtable[i][j] = inputScanner.nextDouble();
                }

        } catch (IOException e) {
            System.out.print(e.getMessage());
        }
    }


    public void initialiseLUT() {
        // @xr add up o(n^4)
        // initiate state-stateIndex table
        int count = 0;
        for (int a = 0; a < numHeading; a++)
            for (int b = 0; b < numEnemyBearing; b++)
                for (int c = 0; c < numEnemyDistance; c++)
                    for (int d = 0; d < numMyEnergy; d++)
                        for (int e = 0; e < numEnemyEnergy; e++)
                            stateIndex[a][b][c][d][e] = count++;

        // initiate stateIndex-action table for QValue
        for (int x = 0; x < numState; x++)
            for (int y = 0; y < numAction; y++)
                // qtable[x][y] = 0;

                // multiplied by 10, since the random is too small, reward is
                // much larger
                qtable[x][y] = 0;
        // qtable[x][y] = Math.random();
    }


    // get stateIndex of given state vector X
    public int indexFor(double[] X) {

        int index = stateIndex[(int) X[0]][(int) X[1]][(int) X[2]][(int) X[3]][(int) X[4]];
        return index;
    }

    // return max QValue
    public double maxQValueFor(int index) {
        double max = 0;
        for (int i = 0; i < numAction; i++) {
            if (qtable[index][i] > max)
                max = qtable[index][i];
        }
        return max;
    }

    // return action with max QValue
    public int actionMaxQValue(int index) {
        double max = 0;
        int action = 0;
        for (int i = 0; i < numAction; i++) {
            if (qtable[index][i] > max) {
                max = qtable[index][i];
                action = i;
            }

        }
        return action;
    }

    public void onBulletHit(BulletHitEvent e) {
        double change = e.getBullet().getPower() * 9;
        reward += change;
        totalReward += change;
    }

    public void onHitByBullet(HitByBulletEvent e) {

        double change = -5 * e.getBullet().getPower();
        reward += change;
        totalReward += change;
    }

    public void onBulletMissed(BulletMissedEvent e) {
        double change = -e.getBullet().getPower();
        reward += change;
        totalReward += change;
    }

    public void onHitRobot(HitRobotEvent e) {

        double change = -6.0;
        reward += change;
        totalReward += change;
    }

    public void onHitWall(HitWallEvent e) {

        double change = -(Math.abs(getVelocity()) * 0.5 - 1);
        reward += change;
        totalReward += change;
        // need more here
    }

    public void onScannedRobot(ScannedRobotEvent e) {

        // the next line gets the absolute bearing to the point where the bot is
        double absbearing_rad = (getHeadingRadians() + e.getBearingRadians())
                % (2 * Math.PI);
        // this section sets all the information about our enemy
        nameEnemy = e.getName();
        double h = nomalizeDegree(e.getHeadingRadians() - headingEnemy);
        h = h / (getTime() - spotTimeEnemy);
        constantHeadingEnemy = h;
        XPositionEnemy = getX() + Math.sin(absbearing_rad) * e.getDistance(); // works
        // out
        // the
        // x
        // coordinate
        // of
        // where
        // the
        // enemy
        // is
        YPositionEnemy = getY() + Math.cos(absbearing_rad) * e.getDistance(); // works
        // out
        // the
        // y
        // coordinate
        // of
        // where
        // the
        // enemy
        // is
        bearingEnemy = e.getBearingRadians();
        headingEnemy = e.getHeadingRadians();
        spotTimeEnemy = getTime(); // game time at which this scan was produced
        speedEnemy = e.getVelocity();
        distanceEnemy = e.getDistance();
        energyEnemy = e.getEnergy();

    }

    public void onWin(WinEvent event) {
//        save(savetest);
        // saveReward();
        battleCount++;
        System.out.println("battleCount");
        totalReward = 0;
    }

    public void onDeath(DeathEvent event) {
//        save(savetest);
        // saveReward();
        battleCount++;
        System.out.println("battleCount");
        totalReward = 0;
    }

    private void saveReward() {
        BufferedWriter writer = null;
        try {
            // ture means append, not clear all
            writer = new BufferedWriter(new FileWriter(rewardCheck, true));
            writer.write(totalReward + "\n");

        } catch (IOException e) {
        } finally {
            try {
                if (writer != null)
                    writer.close();
            } catch (IOException e) {
            }
        }
    }
}
