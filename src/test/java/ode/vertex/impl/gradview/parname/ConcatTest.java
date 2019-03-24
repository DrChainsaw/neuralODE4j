package ode.vertex.impl.gradview.parname;

import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link Concat}
 *
 * @author Christian Skarby
 */
public class ConcatTest {

    /**
     * Test that input is concatenated
     */
    @Test
    public void map() {
        assertEquals("vertex-param", new Concat().map("vertex", "param"));
    }

    /**
     * Test that input is de-concatenated
     */
    @Test
    public void reverseMap() {
        assertEquals(new Pair<>("vertex", "param"), new Concat().reverseMap("vertex-param"));
    }

    /**
     * Test that an error is thrown for illegal input
     */
    @Test(expected = IllegalArgumentException.class)
    public void reverseMapIllegal() {
        new Concat().reverseMap("aaa-vertex-param");
    }
}