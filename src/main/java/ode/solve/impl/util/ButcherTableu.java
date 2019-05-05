package ode.solve.impl.util;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * Struct for representing the Butcher Tableu.
 *
 * @author Christian Skarby
 */
public class ButcherTableu {

    public final INDArray[] a;
    public final INDArray b;
    public final INDArray bStar;
    public final INDArray c;
    public final double[] cMid;

    /**
     * Create a new {@link Builder} instance
     * @return a {@link Builder}.
     */
    public static Builder builder() {
        return new Builder();
    }

    private ButcherTableu(INDArray[] a, INDArray b, INDArray bStar, INDArray c, double[] cMid) {
        this.a = a;
        this.b = b;
        this.bStar = bStar;
        this.c = c;
        this.cMid = cMid;
    }

    /**
     * Builder with built in cache. Why? Because tableus tend to be static and this creates problems in testcases which
     * need to use different data types. Instead of a static tableu, one needs to have a static builder so that a tableu
     * for the correct data type is available.
     */
    public static class Builder {
        final Map<DataType, ButcherTableu> cache = new HashMap<>();
        
        private double[][] a;
        private double[] b;
        private double[] bStar;
        private double[] c;
        private double[] cMid;

        public Builder a(double[][] a) {
            cache.clear();
            this.a = a; return this;
        }

        public Builder b(double[] b) {
            cache.clear();
            this.b = b; return this;
        }

        public Builder bStar(double[] bStar) {
            cache.clear();
            this.bStar = bStar; return this;
        }

        public Builder c(double[] c) {
            cache.clear();
            this.c = c; return this;
        }

        public Builder cMid(double[] cMid) {
            cache.clear();
            this.cMid = cMid; return this;
        }

        public ButcherTableu build() {
            final DataType dataType = Nd4j.defaultFloatingPointType();
            ButcherTableu tableu = cache.get(dataType);
            if(tableu == null) {
                final INDArray[] aArr = new INDArray[a.length];
                for(int i = 0; i < a.length; i++) {
                    aArr[i] = Nd4j.create(a[i]).castTo(dataType).reshape(1, a[i].length);
                }
                tableu = new ButcherTableu(
                        aArr,
                        Nd4j.create(b).castTo(dataType).reshape(1, b.length),
                        Nd4j.create(bStar).castTo(dataType).reshape(1, bStar.length),
                        Nd4j.create(c).castTo(dataType).reshape(1, c.length),
                        cMid);
                cache.put(dataType, tableu);
            }
            return  tableu;
        }
    }
    
}
