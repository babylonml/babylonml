package com.tornadoml.mnist;



public class MNISTBenchLight extends MNISTBase {
    public static void main(String[] args) throws Exception {
        try {
            trainMnist(Integer.MAX_VALUE, 50, 40, 30, 20);
        } catch (Exception e) {
            //noinspection CallToPrintStackTrace
            e.printStackTrace();
            throw e;
        }
    }
}

