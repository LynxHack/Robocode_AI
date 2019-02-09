package NNbot;

import robocode.*;
import robocode.util.Utils;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class FnlBot extends AdvancedRobot {
    // Event reinforcement values, multiplier for adjustment
    public static double reward = 0;
    public static double multiplier = 10; // set to 0 for no intermediate rewards
    public static double termmultiplier = 10; // set to 0 for no terminal rewards
    public static double goodevent = 0.5 * multiplier;
    public static double badevent = -0.5 * multiplier;
    public static double goodend = 2 * termmultiplier;
    public static double badend = -2 * termmultiplier;
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
    public final int neural = 3;
    public int currmethod = sarsa;

    // Params
    public static final double learnRate = 0.1;
    public static final double discountRate = 0.9;
    public static final double epsilon = 0;
    private int laststate;
    private int lastaction;
    public static boolean notinit = true;


    // Save and Load paths
    static File lutfile;    
    static File savefile;
    public static final String filename = "./result.csv";
    public static final String lutfilename = "./qlut.txt";
    public static final String weightfilename = "./weightfile.txt";
    public static final String diffqfilename = "./diffq.txt";

    // Actions
    public static int numAction = 7;
    public final int up = 0;
    public final int down = 1;
    public final int upleft = 2;
    public final int upright = 3;
    public final int downleft = 4;
    public final int downright = 5;
    public final int fire = 6;

    // DEPRECATED
    //    public final int left = 2;
    //    public final int right = 3;
    //    public final int firecannon = 6;

    //DiffQ Observation
    public List<Double> diffQ = new ArrayList<Double>();


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

    //neural network variables
    public int argInputs = 5 + numAction;
    public int argHidden = 12;
    public double learnrate = 0.002;
    public double argMomentum = 0.009;
    public int argA = 0;
    public int argB = 1;
    public boolean bipolar= true;

    NeuralNet NN = new NeuralNet(argInputs, argHidden, learnrate, argMomentum, argA, argB, bipolar);


    // Set whether to load previous qlut table data
    public boolean loadfromfile = false;
    public boolean savelutbool = false;
    public boolean loadweightsfromfile = false;
	public void run() {
	    if(notinit){
            if(currmethod == neural){
                NN.initializeWeights();
                if(loadweightsfromfile){
                    NN.loadWeights(getDataFile(weightfilename));
                    System.out.println("Loaded weights from file");
                }
                System.out.println("Finished creating weights");
                initialiseLUT();
            }
            else{
                System.out.println("LUT method instead of NN");
                if(loadfromfile){
                    File file = getDataFile(lutfilename);
                    loadLUT(file);
                }
                else{
                    initialiseLUT();
                }
            }
            notinit = false;
        }

        lutfile = getDataFile(lutfilename);
        savefile = getDataFile(filename);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForRobotTurn(true);

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

            if(currmethod == neural) {
                System.out.println("Training neural");
                int Action = 0;
                double bestQ = getNNQ(0);
                double newNNQ;
                for (int i = 0; i < numAction; i++) {
                    newNNQ = getNNQ(i);
                    if (newNNQ > bestQ) {
                        Action = i;
                        System.out.println(Action);
                        bestQ = newNNQ;
                    }
                }
                // Train Qlearn or Sarsa in LUTupdate then also use Q values to train Neural network
                LUTupdate(currstate, Action, reward);
                reward = 0;
                setAction(Action);
            }
            else{
                // Train Qlearn or Sarsa in LUTupdate
                LUTupdate(currstate, action, reward);
                reward = 0;
                setAction(action);
            }
        }
    }

    // Removed all distance and coordinate related quantization and trains NN overtime to the Qvalues concurrently
    public double getNNQ(int action){
        double[] X = {2.0 * (double)qheading(RobotHeading)/(double)(numHeading-1) - 1.0 ,
                2.0 * (double) qbearing(EnemyBearing)/(double)(numEnemyBearing-1) - 1.0 ,
                2.0 * (EnemyDistance / 1000.0) - 1,
                2.0 *  (RobotX / 800.0)- 1.0 ,
                2.0 * (RobotY / 600.0) - 1.0
        };

        double[] encodedaction = new double[numAction];
        Arrays.fill(encodedaction, -1.0);
        encodedaction[action] = 1.0;
        X = DoubleStream.concat(Arrays.stream(X), Arrays.stream(encodedaction)).toArray();

//        System.out.println(NN.outputFor(X));
        double y = NN.outputFor(X);
        int currstate = state[qRobotHeading][qEnemyBearing][qEnemyDistance][qx(RobotX)][qy(RobotY)];
        double expectedQval = ((1 - learnRate) * Qold + learnRate * (reward + discountRate * getmax_Q(currstate)) - 10) / 20;
        NN.train(X, expectedQval);

        return NN.outputFor(X);
    }

    //Apply Q-learn algorithm or Sarsa based on currmethod given state, action and accumulated reward
    public void LUTupdate(int state, int action, double reward)
    {
        if(!notinit){
            Qold = qlut[laststate][lastaction];

            if (currmethod == qlearn){

                Qnew = (1 - learnRate) * Qold + learnRate * (reward + discountRate * getmax_Q(state));
                qdiff = Qnew - Qold;
                qlut[laststate][lastaction] = Qnew;
            }
            else {
                Qnew = (1 - learnRate) * Qold + learnRate * (reward + discountRate * qlut[state][action]);
                qdiff = Qnew - Qold;
                qlut[laststate][lastaction] = Qnew;
            }
        }

        // Store Q change for plot
        if(state == 0 && action == 0){
            diffQ.add(qdiff);
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
//        System.out.println("" + maxVal);
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

            case fire:
                setFire(3);
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
            if(numAction == 6){
                setFire(400/e.getDistance()); // Adjust based on distance
            }
//            setFire(2);
        }
//        setTurnGunRightRadians(Utils.normalRelativeAngle((getHeadingRadians() + e
//                .getBearingRadians()) - getGunHeadingRadians())); // move gun toward them

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

         System.out.println("Current win rate out of 100: " + Double.toString(wins/(double) numbattles));

	}

	public void onDeath(DeathEvent event) {
        reward += badend;
        numbattles++;
        losses += 1;
        totalbattles++;

         System.out.println("Current win rate out of 100: " + Double.toString(wins/(double) numbattles));

    }

    public void onBattleEnded(BattleEndedEvent event)
    {
        System.out.println("battle ended");

        //Save stats
        saveResult(savefile);

        //Save LUT
        if(savelutbool) {
            saveLUT(lutfile);
        }

        //Save q differences
        saveQdiff(getDataFile(diffqfilename));
    }

    // Save Q difference stats
    private void saveQdiff(File outputfile)
    {
        PrintStream wStream = null;
        try {
            wStream = new PrintStream(new RobocodeFileOutputStream(outputfile));
            for (int r = 0; r < diffQ.size(); r++) {
                wStream.println(String.format("%s", diffQ.get(r)));
            }
            if (wStream.checkError()) {
                System.err.println("Can not Save!");
            }
            wStream.close();
        } catch (IOException e) {
            System.out.println("IOException: " + e);
        }
        finally {
            try {
                if (wStream != null) {
                    wStream.close();
                }
            } catch (Exception e2) {
                System.out.println("Exception: " + e2);
            }
        }
    }

    // Save winning stats
    private void saveResult(File outputfile)
    {
        System.out.println("Saving Results");
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

    public void saveLUT(File outputfile) {
        PrintStream wStream = null;
        try {
            wStream = new PrintStream(new RobocodeFileOutputStream(outputfile));
            System.out.println(qlut.length);
            System.out.println(qlut[0].length);
            for (int r = 0; r < qlut.length; r++) {
                for (int c = 0; c < qlut[0].length; c++) {
                    float value = (float) qlut[r][c];
                    //shorten decimal places to save memory space
//                    float newval = Math.round(value * 1000) / 1000;
                    DecimalFormat decimalFormat = new DecimalFormat("#.#####");
                    float newval = Float.valueOf(decimalFormat.format(value));

                    if(newval == 0){
                        wStream.println(0);
                    }
                    else{
                        wStream.println(String.format("%s", newval));
                    }
//                    if(newval != 0){
//                        String output = String.format("%d %d %f",r ,c, newval);
//                        wStream.println(output);
//                    }
                }
            }
            if (wStream.checkError()) {
                System.err.println("Can not Save!");
            }
            wStream.close();
        } catch (IOException e) {
            System.out.println("IOException: " + e);
        }
        finally {
            try {
                if (wStream != null) {
                    wStream.close();
                }
            } catch (Exception e2) {
                System.out.println("Exception: " + e2);
            }
        }
    }

    public static double LUTmin = 9999;
	public static double LUTmax = -9999;
    public void loadLUT(File inputFile) {
        BufferedReader rBuffer = null;
        String line;
        try {
            initialiseLUT();
            int count = 1;
            System.out.println("Reading LUT from file");
            rBuffer = new BufferedReader(new FileReader(inputFile));
            for (int r = 0; r < numState; r++) {
                for (int c = 0; c < numAction; c++) {
                    line = rBuffer.readLine();
                    if(line != null){
                        System.out.println(String.format("Line number: %d", count));
                        System.out.println(line);
                        qlut[r][c] = Double.parseDouble(line);
                        if(qlut[r][c] > LUTmax){
                            LUTmax = qlut[r][c];
                        }
                        if(qlut[r][c] < LUTmin){
                            LUTmin = qlut[r][c];
                        }
                        count++;
                    }
                    else{
                        System.out.println(line);
                    }
                }
            }
        }
        catch (NumberFormatException e) {
            System.out.println("error, initializing lut instead");
            initialiseLUT();
        }
        catch (IOException e) {
            System.out.println("IOException: " + e);
        }
        finally {
            try {
                if (rBuffer != null) {
                    rBuffer.close();
                }
            } catch (Exception e2) {
                System.out.println("Exception: " + e2);
            }
        }
    }

    // For training NeuralNetwork Weights
    public static void main(String[] args) throws FileNotFoundException {
        // Start Robot instance to gain access to LUT functions
        FnlBot bot = new FnlBot();
        bot.initialiseLUT();
        File lutfile = new File(lutfilename);
        File weightfile = new File("./bin/NNbot/FnlBot.data/weightfile.txt");
        bot.loadLUT(lutfile);

        List<double[]> inputs = new ArrayList<double[]>();
        List<Double> outputs = new ArrayList<Double>();

        // init state indices for lookup with LUT
        for (int a = 0; a < numHeading; a++){
            for (int b = 0; b < numEnemyBearing; b++){
                for (int c = 0; c < numEnemyDistance; c++){
                    for (int d = 0; d < numX; d++){
                        for (int e = 0; e < numY; e++){
                            for(int action = 0; action < numAction; action++){
                                double[] newinput = {2.0 * (double)a/(double)(numHeading-1) - 1.0 ,
                                        2.0 * (double)b/(double)(numEnemyBearing-1) - 1.0 ,
                                        2.0 * (double)c/(double)(numEnemyDistance-1) -1.0,
                                        2.0 * (double)d/(double)(numX-1) - 1.0 ,
                                        2.0 * (double)e/(double)(numY-1) - 1.0
                                };
                                double[] encodedaction = new double[numAction];
                                Arrays.fill(encodedaction, -1.0);
                                encodedaction[action] = 1.0;

                                //join 2 primitive type array
                                newinput = DoubleStream.concat(Arrays.stream(newinput), Arrays.stream(encodedaction)).toArray();

                                int currstate = state[a][b][c][d][e];
                                double newoutput = (qlut[currstate][action] - -10)/(20);
                                if(qlut[currstate][action] != 0){
                                    inputs.add(newinput);
                                    outputs.add((qlut[currstate][action] - -10)/(20));
                                }

                            }
                        }
                    }
                }
            }
        }

        int count = 0;
        int timesconverged = 0;
        int numberofiterations = 10;
        double totalErr = 0;
        double RMSE = 999999;
        double lowestRMSE = RMSE;

        //Test Constants
        int argInputs = 5 + numAction;
        int argHidden = 20;

        double learnrate = 0.02;
        double argMomentum = 0.09;
        int argA = 0;
        int argB = 1;
        boolean bipolar= true;

        double x[][] = new double[inputs.size()][argInputs];

        double y[] = new double[inputs.size()];
        x = inputs.toArray(x);

        for(int i = 0; i < outputs.size(); i++){
            y[i] = outputs.get(i);
        }

        for(int k =0 ; k< numberofiterations ; k++) {
            NeuralNet NN = new NeuralNet(argInputs, argHidden, learnrate, argMomentum, argA, argB, bipolar);
            NN.initializeWeights();

            int max_epochs = 10000;
            for(int i = 0; i < max_epochs; i++) {
                totalErr = 0;
                for (int j = 0; j < x.length; j++) {
                    totalErr += NN.train(x[j], y[j]);
                }
                RMSE = Math.sqrt(totalErr / 53760);
                // Save onto console with ; delimiter csv format
                System.out.println(i + ";" + RMSE);
                if(RMSE < 0.08){
                    if(RMSE < lowestRMSE){
                        NN.saveWeights(weightfile);
                    }
//                    NN.saveWeights(weightfile);
//                    break;
                }
//                if (0.5 * totalErr < 0.05) {
//                    timesconverged++;
//                    count += i;
//                    break;
//                }
            }
        }
        if(timesconverged > 0) {
            System.out.println("Average epoch: " + count / timesconverged);
        }
        else{
            System.out.println("It never converged!");
        }
    }

}

