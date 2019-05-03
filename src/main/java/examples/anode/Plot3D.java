package examples.anode;

import org.jzy3d.colors.Color;

import java.util.List;

/**
 * Interface for 3D plots. Main use case is to be able to turn off plotting in test cases
 *
 * @author Christian Skarby
 */
interface Plot3D {

    Series3D series(String label);

    void savePicture(String suffix);

    void fit();

    interface Series3D {

        Series3D plot(List<Double> x, List<Double> y, List<Double> z);

        Series3D color(Color color);

        Series3D size(float size);

        Series3D clear();
    }

}
