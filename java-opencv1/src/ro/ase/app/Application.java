package ro.ase.app;

import org.opencv.core.Core;

public class Application {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws InterruptedException {
        // var thread = new Thread(new DetectFaceDemo("best.jpg", "detection.png"));
        // var thread = new Thread(new DetectFaceDemo("faces.jpg", "faces-detection.png"));
        // var thread = new Thread(new DetectFaceDemo("pic1.jpg", "detection2.png"));
        // var thread = new Thread(new DetectFaceDemo("upper_body2.png", "detection_upper.png"));
        var thread = new Thread(new DetectFaceDemo("pic3.jpg", "detection4.png"));
        thread.start();
        thread.join();
    }
}
