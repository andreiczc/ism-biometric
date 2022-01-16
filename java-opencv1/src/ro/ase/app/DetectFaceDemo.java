package ro.ase.app;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

class DetectFaceDemo implements Runnable {
    private final String imgPath;
    private final String outputPath;

    private static final String FACE_XML = "C:\\DevTools\\OpenCV\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml";
    private static final String LOWER_BODY_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_lowerbody.xml";
    private static final String UPPER_BODY_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_upperbody.xml";
    private static final String PROFILE_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_profileface.xml";
    private static final String FULL_BODY_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_fullbody.xml";
    private static final String LEFT_EYE_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_lefteye_2splits.xml";
    private static final String RIGHT_EYE_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml";
    private static final String EARS_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\ears.xml";
    private static final String SMILES_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\haarcascade_smile.xml";
    private static final String NOSE_XML = "C:\\DevTools\\OpenCV\\sources\\data\\haarcascades\\nose.xml";

    public DetectFaceDemo(String imgPath, String outputPath) {
        this.imgPath = imgPath;
        this.outputPath = outputPath;
    }

    public void run() {
        System.out.println("\nRunning DetectFaceDemo");

        Mat image = Imgcodecs.imread(imgPath);

        Mat frameGray = new Mat();
        Imgproc.cvtColor(image, frameGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(frameGray, frameGray);

        MatOfRect faceDetections = detectPart(frameGray, FACE_XML);
        MatOfRect lowerBodyDetections = detectPart(frameGray, LOWER_BODY_XML);
        MatOfRect upperBodyDetections = detectPart(frameGray, UPPER_BODY_XML);
        MatOfRect profileDetections = detectPart(frameGray, PROFILE_XML);
        MatOfRect fullBodyDetections = detectPart(frameGray, FULL_BODY_XML);
        MatOfRect leftEyeDetections = detectPart(frameGray, LEFT_EYE_XML);
        MatOfRect rightEyeDetections = detectPart(frameGray, RIGHT_EYE_XML);
        MatOfRect earDetections = detectPart(frameGray, EARS_XML);
        MatOfRect noseDetections = detectPart(frameGray, NOSE_XML);
        MatOfRect smileDetections = detectPart(frameGray, SMILES_XML, 1.1, 150);

        System.out.printf("Detected %d faces\n", faceDetections.toArray().length);
        System.out.printf("Detected %d lower bodies\n", lowerBodyDetections.toArray().length);
        System.out.printf("Detected %d upper bodies\n", upperBodyDetections.toArray().length);
        System.out.printf("Detected %d profiles\n", profileDetections.toArray().length);
        System.out.printf("Detected %d full bodies\n", fullBodyDetections.toArray().length);
        System.out.printf("Detected %d left eyes\n", leftEyeDetections.toArray().length);
        System.out.printf("Detected %d right eyes\n", rightEyeDetections.toArray().length);
        System.out.printf("Detected %d ears\n", earDetections.toArray().length);
        System.out.printf("Detected %d smiles\n", smileDetections.toArray().length, 1.1, 35);
        System.out.printf("Detected %d noses\n", noseDetections.toArray().length);

        drawRectangle(faceDetections, image, Scalar.all(0));
        drawRectangle(lowerBodyDetections, image, Scalar.all(50));
        drawRectangle(upperBodyDetections, image, Scalar.all(70));
        drawRectangle(profileDetections, image, Scalar.all(90));
        drawRectangle(fullBodyDetections, image, Scalar.all(120));
        drawCircle(leftEyeDetections, image, new Scalar(255, 0, 0));
        drawCircle(rightEyeDetections, image, new Scalar(0, 255, 0));
        drawCircle(earDetections, image, new Scalar(0, 0, 255));
        drawEllipse(smileDetections, image, new Scalar(0, 0, 255));
        drawRectangle(noseDetections, image, new Scalar(0, 0, 255));

        System.out.printf("Writing %s\n", outputPath);
        Imgcodecs.imwrite(outputPath, image);
    }

    private MatOfRect detectPart(Mat image, String pathToXml) {
        return detectPart(image, pathToXml, 1.1, 3);
    }

    private MatOfRect detectPart(Mat image, String pathToXml, double scaleFactor, int neighbours) {
        CascadeClassifier detector = new CascadeClassifier( pathToXml);
        MatOfRect detections = new MatOfRect();
        detector.detectMultiScale(image, detections, scaleFactor, neighbours);

        return detections;
    }

    private void drawRectangle(MatOfRect container, Mat image, Scalar colour) {
        for (Rect rect : container.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), colour);
        }
    }

    private void drawEllipse(MatOfRect container, Mat image, Scalar colour) {
        for (Rect rect : container.toArray()) {
            Imgproc.ellipse(image, new Point( rect.x + rect.width / 2.0, rect.y + rect.height / 2.0),
                    new Size(rect.width / 2.0, rect.height / 2.0),
                    0, 0, 360, colour, 2);
        }
    }

    private void drawCircle(MatOfRect container, Mat image, Scalar colour) {
        for (Rect rect : container.toArray()) {
            Imgproc.circle(image, new Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0),
                    (rect.width + rect.height) / 4, colour);
        }
    }
}