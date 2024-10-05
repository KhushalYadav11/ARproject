package com.arproject;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;
import javafx.application.Platform;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.nio.file.Paths;

public class ARApp extends Application {
    private ImageView imageView;
    private volatile boolean cameraActive = true;
    private Mat glassesOverlay; // New field for glasses overlay

    static {
        try {
            System.out.println("Loading OpenCV library...");
            System.out.println("java.library.path = " + System.getProperty("java.library.path"));
            System.out.println("user.dir = " + System.getProperty("user.dir"));
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            System.out.println("OpenCV library loaded successfully!");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.err.println("Library path: " + System.getProperty("java.library.path"));
            System.err.println("Working directory: " + System.getProperty("user.dir"));
            System.exit(1);
        }
    }

    @Override
    public void start(Stage primaryStage) {
        // Load the overlay image
        String overlayPath = Paths.get("src", "main", "resources", "glasses.png").toString();
        glassesOverlay = Imgcodecs.imread(overlayPath, Imgcodecs.IMREAD_UNCHANGED);

        if (glassesOverlay.empty()) {
            System.err.println("Error: Could not load overlay image from: " + overlayPath);
            return;
        }

        // Rest of your existing start() method code remains the same
        imageView = new ImageView();
        imageView.setFitWidth(800);
        imageView.setFitHeight(600);
        imageView.setPreserveRatio(true);

        StackPane root = new StackPane(imageView);
        Scene scene = new Scene(root, 800, 600);

        primaryStage.setTitle("AR Application");
        primaryStage.setScene(scene);
        primaryStage.show();

        primaryStage.setOnCloseRequest(_ -> {
            cameraActive = false;
            Platform.exit();
            System.exit(0);
        });

        new Thread(this::startCamera).start();
    }

    // New method to overlay images on faces
    private void overlayFace(Mat frame, Mat overlayImage, Rect face) {
        try {
            // Calculate position for overlay (adjust these values as needed)
            int x = face.x;
            int y = face.y;
            int width = face.width;
            int height = face.height / 2; // Glasses usually cover half the face height

            // Resize overlay image to fit the face
            Mat resizedOverlay = new Mat();
            Size size = new Size(width, height);
            Imgproc.resize(overlayImage, resizedOverlay, size);

            // Define region of interest (ROI) in the frame
            Rect roi = new Rect(x, y, width, height);

            // Check if ROI is within frame boundaries
            if (roi.x >= 0 && roi.y >= 0 &&
                    roi.x + roi.width <= frame.cols() &&
                    roi.y + roi.height <= frame.rows()) {

                // Create a Mat for the ROI
                Mat frameROI = frame.submat(roi);

                // Blend the overlay with the frame
                Core.addWeighted(frameROI, 1.0, resizedOverlay, 0.7, 0, frameROI);
            }

            resizedOverlay.release();
        } catch (Exception e) {
            System.err.println("Error overlaying image: " + e.getMessage());
        }
    }

    // Modified startCamera method
    private void startCamera() {
        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Error: Camera is not opened!");
            return;
        }

        Mat frame = new Mat();
        String classifierPath = Paths.get("src", "main", "resources", "haarcascade_frontalface_alt.xml").toString();
        CascadeClassifier faceCascade = new CascadeClassifier(classifierPath);

        if (faceCascade.empty()) {
            System.out.println("Error loading cascade classifier!");
            return;
        }

        File tempFile;
        try {
            tempFile = File.createTempFile("frame_", ".jpg");
            tempFile.deleteOnExit();
        } catch (Exception e) {
            System.out.println("Error creating temp file: " + e.getMessage());
            return;
        }

        while (cameraActive) {
            if (camera.read(frame)) {
                Mat grayFrame = new Mat();
                Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

                // Detect faces
                MatOfRect faceDetections = new MatOfRect();
                faceCascade.detectMultiScale(grayFrame, faceDetections);
                Rect[] faces = faceDetections.toArray();

                // For each detected face, overlay the glasses
                for (Rect face : faces) {
                    overlayFace(frame, glassesOverlay, face);
                }

                final Image imageToShow = convertMatToImage(frame, tempFile);
                Platform.runLater(() -> imageView.setImage(imageToShow));

                grayFrame.release();
                faceDetections.release();
            } else {
                System.out.println("Error: Unable to capture a frame!");
                break;
            }
        }

        frame.release();
        camera.release();
    }

    // Your existing helper methods (convertMatToImage, etc.) remain the same
    private Image convertMatToImage(Mat frame, File tempFile) {
        try {
            Imgcodecs.imwrite(tempFile.getAbsolutePath(), frame);
            return new Image("file:" + tempFile.getAbsolutePath());
        } catch (Exception e) {
            System.out.println("Error converting frame to image: " + e.getMessage());
            return null;
        }
    }

    // Your existing main methods
    public static class Launcher {
        public static void main(String[] args) {
            Application.launch(ARApp.class, args);
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}