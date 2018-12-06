package NNbot;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

import java.io.BufferedReader;
import java.io.FileReader;


public class NeuralNet {
    // NN Constants
    private boolean bipolar;
    private int argNumInputs, argNumHidden, argA, argB;

    // NN Rates
    private double argMomentumTerm;
    private double argRate;

    // Weight Storing Arrays
    private double[][] x2h_w, prev_x2h_w, tempx2h;
    private double[] h2y_w, prev_h2y_w, temph2y;

    private double[] hidden_out, initialHidden;


    /**
     * Constructor. (Cannot be declared in an interface, but your implementation will need one)
     * @param argNumInputs The number of inputs in your input vector
     * @param argNumHidden The number of hidden neurons in your hidden layer. Only a single hidden layer is supported
     * @param argLearningRate The learning rate coefficient
     * @param argMomentumTerm The momentum coefficient
     * @param argA Integer lower bound of sigmoid used by the output neuron only.
     * @param argB Integer upper bound of sigmoid used by the output neuron only.
     **/

    public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate,double argMomentumTerm, int argA, int argB, boolean bipolar) {
        //Initialize Constants
        this.argA = argA;
        this.argB = argB;
        this.argNumInputs = argNumInputs;
        this.argNumHidden = argNumHidden;

        // Part 3
        this.argRate = argLearningRate;
        this.argMomentumTerm = argMomentumTerm;
        this.bipolar = bipolar;

        //Initialize Weights for Synapses, add one to size for biases
        x2h_w = new double[argNumHidden][argNumInputs+1];
        h2y_w = new double[argNumHidden+1];

        //Save Previous Weights , initialize them first to 0 then update later
        hidden_out = new double[argNumHidden];
        prev_x2h_w = new double[argNumHidden][argNumInputs+1];
        prev_h2y_w = new double[argNumHidden+1];

        //Temporary Arrays for Swapping
        initialHidden = new double[argNumHidden];
        tempx2h = new double[argNumHidden][argNumInputs+1];
        temph2y = new double[argNumHidden+1];
        //zeroWeights();
    }

    /**
     * Returns bipolar sigmoid for input x
     * @param x: Input
     * @return f(x) = 2 / (1+exp(-x)) - 1
     */
    public double sigmoid(double x) {
        return 2 * ((argB - argA)/(1 + Math.exp(-x)) + argA) - 1;
    }

    /**
     * Method implements general sigmoid asymtote bound (a,b)
     * @param x: Input
     * @return f(x) = b_minus_a / (1+exp(-x)) - minus_a
     */
    public double customSigmoid(double x) {
        return (argB - argA)/(1 + Math.exp(-x)) + argA;
    }

    /**
     * Initialize weights with for nodes (randomized)
     * For say 2 inputs, the input vector is [0] & [1]. We add [2] for the bias.
     * Like wise for hidden units. For say 2 hidden units which are stored in an array.
     * [0] & [1] are the hidden & [2] the bias.
     * We also initialise the last weight change arrays. This is to implement the alpha term.
     */
    public void initializeWeights() {
        Random randweight = new Random();
        for(int i=0; i<argNumHidden; i++) {
            for(int j=0; j <argNumInputs; j++) {
                x2h_w[i][j] = randweight.nextDouble() - 0.5;
                prev_x2h_w[i][j] = x2h_w[i][j];
            }
            x2h_w[i][argNumInputs] = randweight.nextDouble () - 0.5;
            prev_x2h_w[i][argNumInputs] = x2h_w[i][argNumInputs];

        }

        for(int i=0; i<argNumHidden; i++) {
            h2y_w[i] = randweight.nextDouble() - 0.5;
            prev_h2y_w[i] = h2y_w[i];
        }
        h2y_w[argNumHidden] = randweight.nextDouble () - 0.5;
        prev_h2y_w[argNumHidden] = h2y_w[argNumHidden];
    }

    /**
     * Initialize weights to 0
     */
    public void zeroWeights() {
        for(int i=0; i<argNumHidden; i++)
            for(int j=0; j < argNumInputs + 1; j++) {
                prev_x2h_w[i][j] = 0;
            }

        for(int i=0; i<=argNumHidden; i++) {
            prev_h2y_w[i] = 0;
        }
    }

    /**
     *
     * @param X: The input Vector. double array.
     * @return Value returned by th LUT or NN for input vector
     */
    public double outputFor(double[] X) {
        for(int i=0; i<argNumHidden; i++) {
            initialHidden[i] = x2h_w[i][argNumInputs]*1.0;
            for(int j=0; j < argNumInputs; j++) {
                initialHidden[i] += x2h_w[i][j] * X[j];
            }
            hidden_out[i] = customSigmoid(initialHidden[i]);
            if(bipolar){
                hidden_out[i] = sigmoid(initialHidden[i]);
            }
        }

        double y = 0;
        for(int i=0; i<argNumHidden; i++) {
            y +=  ( h2y_w[i] * hidden_out[i] );
        }

        y += h2y_w[argNumHidden] * 1.0;

        //Bipolar -1 to 1 squish
        if(bipolar){
            return sigmoid(y);
        }
        return customSigmoid(y);
    }

    /**
     * Method tells NN or LUT output value to map to input vector
     * ex. The desired correct output value for given input
     * @param X: The input vector
     * @param argValue: The new value to learn
     * @return The error of output for input vector
     */
    public double train(double[] X, double argValue) {
        //2a - FWD PROP
        double y = outputFor(X);
        double fprime = y * (1 - y);

        //System.out.println("predict: " + y);

        // For bipolar
        if(bipolar) {
            fprime = 0.5 * (1 - Math.pow(y, 2));
        }

        //2b - BWD PROP
        double outErr = (argValue - y) * fprime;

        double hiddenErr[] = new double[argNumHidden];
        for(int i = 0; i < argNumHidden; i++) {
            hiddenErr[i] = h2y_w[i] * outErr * (hidden_out[i] - argA) * (argB - hidden_out[i])/(argB - argA);
        }

        //2c - Weight Updates
        // Update hidden to output synapse weights
        System.arraycopy(h2y_w, 0, temph2y, 0, h2y_w.length);
        for(int i = 0; i < argNumHidden; i++) {
            h2y_w[i] += (argRate * outErr * hidden_out[i]) + (h2y_w[i] - prev_h2y_w[i]) * argMomentumTerm;
        }
        // Update Bias
        h2y_w[argNumHidden] += (argRate * outErr * 1)
                +(h2y_w[argNumHidden] - prev_h2y_w[argNumHidden])
                * argMomentumTerm;
        System.arraycopy(temph2y, 0, prev_h2y_w, 0, temph2y.length);


        // Update input to hidden synapse weights
        for (int i = 0; i < argNumHidden; i++){
            System.arraycopy( x2h_w[i], 0, tempx2h[i], 0, x2h_w[i].length );
        }

        for(int i = 0; i < argNumHidden; i++) {
            for(int j = 0; j < argNumInputs; j++) {
                x2h_w[i][j] += (argRate * hiddenErr[i] * X[j])
                        + (x2h_w[i][j] - prev_x2h_w[i][j])
                        * argMomentumTerm;
            }

            // Update Biases
            x2h_w[i][argNumInputs] += (argRate * hiddenErr[i] * 1)
                    + (x2h_w[i][argNumInputs] - prev_x2h_w[i][argNumInputs])
                    * argMomentumTerm;
        }

        // Update input to hidden synapse weights
        for (int i = 0; i < argNumHidden; i++){
            System.arraycopy( tempx2h[i], 0, prev_x2h_w[i], 0, tempx2h[i].length );
        }

        // Return LS error
        return Math.pow((y - argValue), 2);
    }



//    /**
//     * Write either LUT or weights to for the NN to a file
//     * @param argFile: type file input
//     */
//    public void save(File argFile)  throws FileNotFoundException{
//        PrintWriter pw = new PrintWriter(argFile);
//        StringBuilder builder = new StringBuilder();
//        String ColumnNamesList = "Node, Weights";
//
//        for(int i = 0; i < argNumInputs + 1; i++){
//            for(int j = 0; j < argNumHidden; j++){
//                builder.append(x2h_w[j][i]);
//                builder.append("\n");
//            }
//        }
//
//        //Neuron Layer Delimiter for Loading Purposes
//        builder.append("-");
//
//        for(int i = 0; i < argNumHidden; i++){
//            builder.append(h2y_w[i]);
//            builder.append("\n");
//        }
//
//        builder.append("*");
//        pw.write(builder.toString());
//        pw.close();
//    }
//
//    /**
//     * Loads LUT / NN weights from given file. Load have knowledge of how data
//     * was written by save method. Raise error when trying to load incorrect format
//     * ex. Wrong number of neurons
//     * @param argFileName
//     * @throws IOException
//     */
//
//    public void load(String argFileName) throws IOException {
//        String csvFile = argFileName;
//        String line = "";
//        String cvsSplitBy = ",";
//
//        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
//            int weightcount = 0;
//            int rownum = 0;
//            int colnum = 0;
//            while ((line = br.readLine()) != "-") {
//                // use comma as separator
//                String[] node_info = line.split(cvsSplitBy);
//                weightcount += 1;
//                System.out.println("[Node Number: " + node_info[0] + " , Weight =" + node_info[1]);
//
//                if(weightcount <= argNumHidden * (argNumInputs + 1)){
//                    rownum = (int)(Math.floor(weightcount /(argNumHidden)));
//                    colnum = weightcount % argNumHidden;
//                    x2h_w[rownum][colnum] = Double.parseDouble(node_info[1]);
//                }
//                else{
//                    throw new Error("Incorrect neuron dimension!");
//                }
//            }
//
//            weightcount = 0;
//            while ((line = br.readLine()) != "-") {
//                // use comma as separator
//                String[] node_info = line.split(cvsSplitBy);
//                weightcount += 1;
//                System.out.println("[Node Number: " + node_info[0] + " , Weight =" + node_info[1]);
//
//                if(weightcount <= argNumHidden){
//                    h2y_w[weightcount] = Double.parseDouble(node_info[1]);
//                }
//                else{
//                    throw new Error("Incorrect neuron dimension!");
//                }
//            }
//
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }
}

