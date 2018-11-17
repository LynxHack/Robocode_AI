package fnl;

import java.awt.*;
import java.awt.geom.*;

import java.io.*;
import robocode.*;
import robocode.AdvancedRobot;
import robocode.ScannedRobotEvent;
import robocode.util.Utils;

public class FnlBot extends AdvancedRobot {
    // Event reinforcement values, multiplier for adjustment
    public static double reward = 0;
    public static double multiplier = 1; // set to 0 for no intermediate rewards
    public static double termmultiplier = 1; // set to 0 for no terminal rewards
    public static double goodevent = 5 * multiplier;
    public static double badevent = -5 * multiplier;
    public static double goodend = 20 * termmultiplier;
    public static double badend = -20 * termmultiplier;
    private double Qold;
    private double Qnew;

    public static int numbattles = 1;
    public static int totalbattles = 1;
    public static double wins = 0;
    public static double losses = 0;

    private static final int numrounds = 100000;
    private static int[] numwins = new int[numrounds / 100];

    // Learning Method
    public final int random = 0;
    public final int qlearn = 1;
    public final int sarsa = 2;
    public int currmethod = qlearn;

    // Params
    public static final double learnRate = 0.1;
    public static final double discountRate = 0.9;
    public static final double epsilon = 0.7;
    private int laststate;
    private int lastaction;
    public static boolean notinit = true;


    // Save and Load paths
    static File lutfile;    
    static File savefile;
    private static final String filename = "./e07.csv";
    private static final String lutfilename = "./qlut.csv";

    // Actions
    public static int numAction = 6;
    public final int up = 0;
    public final int down = 1;
    public final int upleft = 2;
    public final int upright = 3;
    public final int downleft = 4;
    public final int downright = 5;

    // DEPRECATED
    //    public final int left = 2;
    //    public final int right = 3;
    //    public final int firecannon = 6;


    public double movedist = 200;
    public double turnangle = 45;

    // Set default firepower to simplify code
    public int firepower = 1;
    
    // States
    // in degrees, //0 - 90, //91 - 180, //181 - 270, //271 - 360
    public static final int numHeading = 4;
    // in degrees, //0 - 90, //91 - 180, //181 - 270, //271 - 360
    public static final int numEnemyBearing = 4;
    // Field dimension 600 * 800 => max dist = 1000. Thus, divide into segments of 100s
    public static final int numEnemyDistance = 10;
    // 10 possible distances in increments of 100 for 0 - 800;
    public static final int numX = 8;
    // 10 possible distances in increments of 100 for 0 - 600;
    public static final int numY = 6;


    public static int state[][][][][] = new int[numHeading]
                                        [numEnemyBearing]
                                        [numEnemyDistance]
                                        [numX]
                                        [numY];

    private static final int numState = numHeading
                                        * numEnemyBearing
                                        * numEnemyDistance
                                        * numX
                                        * numY;

    static double[][] qlut = new double[numState][numAction];
    static double qdiff;

    // Robot Tracking
    // Variables to track the state of the arena
    public int RobotX;
    public int RobotY;
    public int RobotHeading;
    public int RobotGunHeading;
    public int RobotGunBearing;
    public int RobotEnergy;
    public int EnemyDistance;
    public int EnemyHeading;
    public int EnemyBearing;
    public int EnemyBearingFromGun;
    public int EnemyEnergy;
    public int absoluteBearing;
    public int EnemyX;
    public int EnemyY;
    public int EnemyVelocity;
    public long EnemyTime;

    public int qRobotHeading;
    public int qRobotGunHeading;
    public int qRobotGunBearing;
    public int qRobotEnergy;
    public int qEnemyDistance;
    public int qEnemyHeading;
    public int qEnemyBearing;
    public int qEnemyBearingFromGun;
    public int qEnemyEnergy;


	public void run() {

	    if(notinit){
            initialiseLUT();
            notinit = false;
        }
        else{
//            System.out.println(lutfile);
//            loadLUT(lutfile);
        }
        lutfile = getDataFile(filename);
        savefile = getDataFile(filename);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);

        if (getGunHeat() == 0) {
            setFire(1);
        }

		while(true) {
		    doAction();
            if ( getRadarTurnRemaining() == 0.0 )
                setTurnRadarRightRadians( Double.POSITIVE_INFINITY );
            setTurnGunRight(getHeading() - getGunHeading() + EnemyBearing);
            execute();

		}
	}

    public void initialiseLUT() {
        // init LUT with 0s
        for (int x = 0; x < numState; x++)
            for (int y = 0; y < numAction; y++)
                qlut[x][y] = 0;

		// init state indices for lookup with LUT
		int count = 0;
        for (int a = 0; a < numHeading; a++)
            for (int b = 0; b < numEnemyBearing; b++)
                for (int c = 0; c < numEnemyDistance; c++)
                    for (int d = 0; d < numX; d++)
                        for (int e = 0; e < numY; e++)
                            state[a][b][c][d][e] = count++;
	}

	// Perform either randomized no learning method, qlearning / sarsa with exploration rate settable from 0 (no exploration) to 1
	public void doAction(){
        if(currmethod == random){
            int action = (int) Math.floor(Math.random() * numAction); // pick from 4 possible actions of up down left right
            setAction(action);
        }
        else if(currmethod != random){
            // Check if explore or not with epsilon
            int action = 0;
            int currstate = state[qRobotHeading][qEnemyBearing][qEnemyDistance][qx(RobotX)][qy(RobotY)];

            if(Math.random() < epsilon){
                action = (int) Math.floor(Math.random() * numAction); // pick from 6 possible actions
            }
            else{
                action = pickBestAction(currstate);
            }
            setAction(action);

            // Train Qlearn or Sarsa in LUTupdate
            LUTupdate(currstate, action, reward);
        }
    }

    //Apply Q-learn algorithm or Sarsa based on currmethod given state, action and accumulated reward
    public void LUTupdate(int state, int action, double reward)
    {
        if(!notinit){
            Qold = qlut[laststate][lastaction];

            if (currmethod == qlearn){
                Qnew = (1 - learnRate) * Qold + learnRate * (reward + discountRate * getmax_Q(state));
//            qdiff = learnRate * (reward + discountRate * getmax_Q(state) - Qprev);
            }
            else {
                Qnew = (1 - learnRate) * Qold + learnRate * (reward + discountRate * qlut[state][action]);
                qdiff = Qnew - Qold;
                qlut[laststate][lastaction] = Qnew;
            }
        }

        laststate = state;
        lastaction = action;

    }

    // Pick action of best qval from lut
    public int pickBestAction(int state)
    {
        double maxVal = Double.NEGATIVE_INFINITY;
        int bestAction = 0;
        for (int i = 0; i < qlut[state].length; i++)
        {
            double qval = qlut[state][i];
            if (qval > maxVal)
            {
                maxVal = qval;
                bestAction = i;
            }
        }

//        System.out.println("Selected action is " + bestAction);
        return bestAction;
    }

    // Find the highest Qval
    public double getmax_Q(int state)
    {
        double maxVal = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < qlut[state].length; i++)
        {
            if (qlut[state][i] > maxVal){
                maxVal = qlut[state][i];
            }
        }
        return maxVal;
    }

    public void setAction(int action){
        switch (action) {
            case up:
                setAhead(movedist);
                break;

            case down:
                setBack(movedist);
                break;

            case upleft:
                setAhead(movedist);
                setTurnLeft(turnangle);
                break;

            case upright:
                setAhead(movedist);
                setTurnRight(turnangle);
                break;

            case downleft:
                setBack(movedist);
                setTurnRight(turnangle);
                break;

            case downright:
                setBack(movedist);
                setTurnRight(turnangle);
                break;
        }
    }

    // Wide angle scanning method - courtesy of Robowiki tutorials
    public void radarlock(ScannedRobotEvent e){
        double angleToEnemy = getHeadingRadians() + e.getBearingRadians();
        double radarTurn = Utils.normalRelativeAngle( angleToEnemy - getRadarHeadingRadians() );
        double extraTurn = Math.min( Math.atan( 36.0 / e.getDistance() ), Math.PI / 8 );

        if (radarTurn < 0)
            radarTurn -= extraTurn;
        else
            radarTurn += extraTurn;

        setTurnRadarRightRadians(radarTurn);
    }

    public void onScannedRobot(ScannedRobotEvent e){
        radarlock(e);
        if(getGunTurnRemaining() < 10){
//            setFire(400 / e.getDistance()); // Adjust based on distance
            setFire(2);
        }

        // Our info
        RobotX = (int) getX();
        RobotY = (int) getY();
        RobotHeading = (int) getHeading();
        RobotGunHeading = (int) getGunHeading();
        RobotGunBearing = normalizeBearing(RobotHeading - RobotGunHeading);
        RobotEnergy = (int) getEnergy();
        
        // Enemy's info
        absoluteBearing = (int) (getHeadingRadians() + e.getBearingRadians());
        EnemyDistance = (int) e.getDistance();
        EnemyHeading = (int) e.getHeading();
        EnemyBearing = (int) e.getBearing();
        EnemyBearingFromGun = normalizeBearing(RobotGunBearing + EnemyBearing);
        EnemyEnergy = (int) e.getEnergy();

        EnemyX = (int) (getX() + e.getDistance() * Math.sin(absoluteBearing));
        EnemyY = (int) (getY() + e.getDistance() * Math.cos(absoluteBearing));
        EnemyVelocity = (int) e.getVelocity();
        EnemyTime = getTime();

        // Quantize
        qRobotHeading = qheading(RobotHeading);
        qRobotGunHeading = qheading(RobotGunHeading);
        qRobotGunBearing = qbearing(RobotGunBearing);
        qRobotEnergy = qenergy(RobotEnergy);
        qEnemyDistance = qdist(EnemyDistance);
        qEnemyHeading = qheading(EnemyHeading);
        qEnemyBearing = qbearing(EnemyBearing);
        qEnemyBearingFromGun = qbearing(EnemyBearingFromGun);
        qEnemyEnergy = qenergy(EnemyEnergy);
    }

    //Convert the x into 10 outputs
    public static int qx(int arg) {
        int dist=(int) Math.floor(arg / 100);
        //System.out.println("X is "  + Integer.toString((dist)));
        return dist;
    }

    //Convert the y into 10 outputs
    public static int qy(int arg) {
        int dist=(int) Math.floor(arg / 100);
        //System.out.println("Y is "  + Integer.toString((dist)));
        return dist;
    }

    //Convert the distance into 10 outputs
    public static int qdist(int value)
    {
        int dist=(int) Math.floor(value / 100);
        //System.out.println("distance is "  + Integer.toString((dist)));
        return dist;
    }


	public int qenergy(int energy){
	    if(energy <= 30){
	        return 0;
        }
        else if(energy <= 71){
            return 1;
        }
        else{
            return 2;
        }
    }

    // Convert heading into 4 possible outputs
    public int qheading(int heading){
        if(heading <= 90){
            return 0;
        }
        else if(heading <= 180){
            return 1;
        }
        else if(heading <= 270){
            return 2;
        }
        else {
            return 3;
        }
    }

    // Convert bearing into 4 possible outputs
    public int qbearing(int bearing){
        if(bearing <= -90){
            return 0;
        }
        else if(bearing <= 0){
            return 1;
        }
        else if(bearing <= 90){
            return 2;
        }
        else {
            return 3;
        }
    }

    // return a positive value of enemy angle
    public int normalizeBearing(int angle){
        int result = angle;
        while (result > 180) {
            result -= 360;
        }
        while (result < -180) {
            result += 360;
        }

        return result;
    }

    // Intermediate Events have heavy emphasis on dodging bullets
    public void onHitByBullet(HitByBulletEvent e) 
    {
        reward +=  2 * badevent;
    }
   
    public void onBulletHit(BulletHitEvent e)
    {
        reward +=  goodevent;
    }

    public void onBulletMissed(BulletMissedEvent e) 
    { 
        reward += badevent / 2;
    }

    public void onHitWall(HitWallEvent e) 
    {
        reward += badevent / 2;
    } 
   
    // Terminal Events
    public void onWin(WinEvent event) {
        reward += goodend;
        numbattles++;
        wins += 1;
        numwins[(getRoundNum() - 1) / 100] += 1;
        totalbattles++;

        // System.out.println("Current win rate out of 100: " + Double.toString(wins/(double) numbattles));

	}

	public void onDeath(DeathEvent event) {
        reward += badend;
        numbattles++;
        losses += 1;
        totalbattles++;

        // System.out.println("Current win rate out of 100: " + Double.toString(wins/(double) numbattles));

    }

    public void onBattleEnded(BattleEndedEvent event)
    {
        saveResult(savefile);
//        saveLUT(lutfile);
    }

    // Save winning stats
    private void saveResult(File outputfile)
    {
        try
        {
            RobocodeFileOutputStream fileOut = new RobocodeFileOutputStream(outputfile);
            PrintStream out = new PrintStream(new BufferedOutputStream(fileOut));
            out.format("Learning Rate, %f,\n", learnRate);
            out.format("Discount Rate, %f,\n", discountRate);
            out.format("Explore Rate, %f,\n", epsilon);
            if (currmethod == random){
                out.format("Method, random,\n");
            }
            else if(currmethod == qlearn){
                out.format("Method, qlearn, \n");
            }
            else if(currmethod == sarsa){
                out.format("Method, sarsa, \n");
            }

            out.format("Per 100 Rounds, # Wins,\n");
            for (int i = 0; i < getRoundNum()/100; i++)
            {
                out.format("%d, %d,\n", i + 1, numwins[i]);
            }

            out.close();
            fileOut.close();
        }
        catch (IOException e)
        {
            System.out.print("ERROR in save result!");
            e.printStackTrace();
        }
    }

    // Save and Reload LUT data
    public void loadLUT(File file)
    {
        try
        {
            BufferedReader loadfile;
            loadfile = new BufferedReader(new FileReader(file));
            for (int i = 0; i < numState; i++)
                for (int j = 0; j < numAction; j++)
                    qlut[i][j] = Double.parseDouble(loadfile.readLine());
        }
        catch (IOException e)
        {
            System.out.println("Could not read from file! " + e);
        }
    }

    public void saveLUT(File file)
    {
        try
        {
            PrintStream savefile = new PrintStream(new RobocodeFileOutputStream(file));
            for (int i = 0; i < numState; i++)
                for (int j = 0; j < numAction; j++)
                    savefile.println(new Double(qlut[i][j]));

            savefile.close();
        }
        catch (IOException e)
        {
            System.out.println("Error trying to save LUT! " + e);
        }

    }
}

