package com.babylonml.backend.gemm;

import uk.ac.manchester.tornado.api.WorkerGrid2D;

public class WorkerGrid2DFixed extends WorkerGrid2D {
    public WorkerGrid2DFixed(int x, int y) {
        super(x, y);
    }

    public void setNumberOfWorkgroups(int x, int y) {
        this.numOfWorkgroups = new long[]{x, y, 1};
    }
}
