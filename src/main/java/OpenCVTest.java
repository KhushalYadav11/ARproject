import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfRect;

public class OpenCVTest {

    static {
        try {
            System.loadLibrary("opencv_java490");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load: " + e);
            System.err.println("Library path: " + System.getProperty("java.library.path"));
        }
    }

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("Error: Camera is not opened!");
            return;
        }

        Mat frame = new Mat();
        CascadeClassifier faceCascade = new CascadeClassifier("C:\\Users\\hp\\IdeaProjects\\AR_project\\src\\main\\resources\\haarcascade_frontalface_alt.xml");
        if (faceCascade.empty()) {
            System.out.println("Error loading cascade classifier!");
            return;
        }

        Mat overlayImage = Imgcodecs.imread("C:\\Users\\hp\\IdeaProjects\\AR_project\\src\\main\\resources\\overlay_image.jpg");
        if (overlayImage.empty()) {
            System.out.println("Error loading overlay image!");
            return;
        }

        while (true) {
            if (camera.read(frame)) {
                Mat grayFrame = new Mat();
                Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

                // Detect faces
                Rect[] faces = detectFaces(grayFrame, faceCascade);
                for (Rect face : faces) {
                    // Draw rectangle around detected face
                    Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);

                    // Overlay the image on top of the detected face
                    overlayFace(frame, overlayImage, face);
                }

                // Display the frame (You may want to use a GUI library for display)
                Imgcodecs.imwrite("output_frame.jpg", frame);
                System.out.println("Frame processed and saved as output_frame.jpg");

                // Sleep briefly to manage CPU usage
                try {
                    Thread.sleep(33); // Roughly 30 FPS
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                System.out.println("Error: Unable to capture a frame!");
                break;
            }
        }

        camera.release();
    }

    private static Rect[] detectFaces(Mat grayFrame, CascadeClassifier faceCascade) {
        MatOfRect faceDetections = new MatOfRect();
        faceCascade.detectMultiScale(grayFrame, faceDetections);
        return faceDetections.toArray();
    }

    private static void overlayFace(Mat frame, Mat overlayImage, Rect face) {
        // Calculate the position for the overlay image
        int x = face.x;
        int y = face.y;
        int width = face.width;
        int height = face.height;

        // Resize overlay image to fit the detected face
        Mat resizedOverlay = new Mat();
        Imgproc.resize(overlayImage, resizedOverlay, new Size(width, height));

        // Overlay the image
        resizedOverlay.copyTo(frame.submat(y, y + height, x, x + width));
    }
}
