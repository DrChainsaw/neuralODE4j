package ode.solve.impl.util;

import org.nd4j.linalg.api.buffer.DataBuffer;
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

    private ButcherTableu(INDArray[] a, INDArray b, INDArray bStar, INDArray c) {
        this.a = a;
        this.b = b;
        this.bStar = bStar;
        this.c = c;
    }

    /**
     * Builder with built in cache. Why? Because tableus tend to be static and this creates problems in testcases which
     * need to use different data types. Instead of a static tableu, one needs to have a static builder so that a tableu
     * for the correct data type is available.
     */
    public static class Builder {
        final Map<DataBuffer.Type, ButcherTableu> cache = new HashMap<>();
        
        private double[][] a;
        private double[] b;
        private double[] bStar;
        private double[] c;

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

        public ButcherTableu build() {
            ButcherTableu tableu = cache.get(Nd4j.dataType());
            if(tableu == null) {
                final INDArray[] aArr = new INDArray[a.length];
                for(int i = 0; i < a.length; i++) {
                    aArr[i] = Nd4j.create(a[i]);
                }
                tableu = new ButcherTableu(
                        aArr,
                        Nd4j.create(b),
                        Nd4j.create(bStar),
                        Nd4j.create(c));
                cache.put(Nd4j.dataType(), tableu);
            }
            return  tableu;
        }
    }
    
}
