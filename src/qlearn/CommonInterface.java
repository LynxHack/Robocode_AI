package qlearn;

import java.io.*;

public interface CommonInterface {
    public double outputFor(double [] X);
    public double train(double [] X, double argValue);
    public void save(File argFile);
    public void load(File argFile) throws IOException;
}
