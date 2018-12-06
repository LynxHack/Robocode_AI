package NNbot;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class Offlinetrain {
    public static void main(String[] args) throws FileNotFoundException {
        //PrintWriter pw = new PrintWriter(new File("test.csv"));
        //StringBuilder builder = new StringBuilder();
        //String ColumnNamesList = "Epoch, TotalLSE";

        int count = 0;
        int timesconverged = 0;
        int numberofiterations = 500;
        double totalErr = 0;
        //Test Constants
        int argInputs = 2;
        int argHidden = 4;
        double learnrate = 0.2;
        double argMomentum = 0.9;
        int argA = 0;
        int argB = 1;
        boolean bipolar= true;

        //Unipolar test
//             double x[][] = {{0,0}, {0,1}, {1,0}, {1,1}};
//             double y[] = {0,1,1,0};

        // For bipolar tests
        double x[][] = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
        double y[] = {-1,1,1,-1};


        for(int k =0 ; k< numberofiterations ; k++) {
            NeuralNet XOR = new NeuralNet(argInputs, argHidden, learnrate, argMomentum, argA, argB, bipolar);
            XOR.initializeWeights();

            int max_epochs = 10000;
            for(int i = 0; i < max_epochs; i++) {
                totalErr = 0;
                for (int j = 0; j < x.length; j++) {
                    totalErr += XOR.train(x[j], y[j]);
                }

                System.out.println("Epoch " + i + " error " + totalErr * 0.5);
                //builder.append(i + ",");
                //builder.append(totalErr * 0.5);
                //builder.append("\n");
                //System.out.println(k+1 +" "+ i +" "+ totalErr * 0.5);

                if (0.5 * totalErr < 0.05) {
                    timesconverged++;
                    count += i;
                    break;
                }
            }
        }
        if(timesconverged > 0) {
            System.out.println("Average epoch: " + count / timesconverged);
        }
        else{
            System.out.println("It never converged!");
        }

        //pw.write(builder.toString());
        //pw.close();
    }
}
