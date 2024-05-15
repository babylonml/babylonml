package com.tornadoml.mnist;


public class MNISTBench extends MNISTBase {
    public static void main(String[] args) throws Exception {
        try {
            trainMnist(Integer.MAX_VALUE, 2500, 2000, 1500, 1000, 500);
        } catch (Exception e) {
            //noinspection CallToPrintStackTrace
            e.printStackTrace();
            throw e;
        }
    }
}

