package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.WorkerGrid;

public class ExplicitWorkerGrid2D implements WorkerGrid {
//    private long[] globalWork;
//    private long[] numOfWorkgroups;
//    private long[] localWork;
//    private long[] globalOffset;
//
//
//    public ExplicitWorkerGrid2D(long x, long y) {
//        this.numOfWorkgroups = new long[]{x, y, 1};
//        this.globalOffset = new long[]{0, 0, 0};
//    }
//
//    @Override
//    public int dimension() {
//        return 2;
//    }
//
//    @Override
//    public long[] getGlobalWork() {
//        return globalWork;
//    }
//
//    @Override
//    public long[] getLocalWork() {
//        return localWork;
//    }
//
//    @Override
//    public long[] getNumberOfWorkgroups() {
//        return numOfWorkgroups;
//    }
//
//    @Override
//    public long[] getGlobalOffset() {
//        return globalOffset;
//    }
//
//    @Override
//    public void setGlobalWork(long x, long y, long z) {
//        this.globalWork = new long[]{x, y, z};
//    }
//
//    @Override
//    public void setLocalWork(long x, long y, long z) {
//        this.localWork = new long[]{x, y, z};
////        calculateNumberOfWorkgroups();
//
//    }
//
//    @Override
//    public void setNumberOfWorkgroupsToNull() {
//
//    }
//
//    @Override
//    public void setLocalWorkToNull() {
//
//    }
//
//    @Override
//    public void setGlobalOffset(long x, long y, long z) {
//
//    }


    @Override
    public int dimension() {
        return 0;
    }

    @Override
    public long[] getGlobalWork() {
        return new long[0];
    }

    @Override
    public long[] getLocalWork() {
        return new long[0];
    }

    @Override
    public long[] getNumberOfWorkgroups() {
        return new long[0];
    }

    @Override
    public long[] getGlobalOffset() {
        return new long[0];
    }

    @Override
    public void setGlobalWork(long x, long y, long z) {

    }

    @Override
    public void setLocalWork(long x, long y, long z) {

    }

    @Override
    public void setNumberOfWorkgroupsToNull() {

    }

    @Override
    public void setLocalWorkToNull() {

    }

    @Override
    public void setGlobalOffset(long x, long y, long z) {

    }
}
