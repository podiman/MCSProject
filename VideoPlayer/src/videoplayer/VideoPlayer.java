/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package videoplayer;

import java.awt.AWTException;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.File;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.stage.Stage;
import javax.imageio.ImageIO;
import Luxand.*;
import Luxand.FSDK.*;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.io.BufferedReader;
import java.util.ArrayList;
import javafx.concurrent.Task;
import javafx.concurrent.ScheduledService;
import javafx.util.Duration;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 *
 * @author mayurawijeyaratne
 */
public class VideoPlayer extends Application {

    private static final double FRAME_RATE = 50;
    private static final int SECONDS_TO_RUN_FOR = 1;
    public static final double SECONDS_BETWEEN_FRAMES = 10;
    private Toolkit toolkit;
    private static Rectangle screenBounds;
    public final java.util.List<TFaceRecord> FaceList = new ArrayList<TFaceRecord>();
    private static int i;
    FileWriter writer = null;
    String sFileName = "/Users/mayurawijeyaratne/NetBeansProjects/VideoPlayer/test.csv";

    private static final String outputFilePrefix = "G:\\MCS\\VideoPlayer\\snapshots\\";

    public class TFaceRecord {

        public FSDK.FSDK_FaceTemplate.ByReference FaceTemplate;
        public FSDK.TFacePosition.ByReference FacePosition;
        public FSDK.FSDK_Features.ByReference FacialFeatures;
        public String ImageFileName;
        public FSDK.HImage image;
        public FSDK.HImage faceImage;
    }

    @Override
    public void start(Stage stage) throws Exception {

        try {
            int r = FSDK.ActivateLibrary("IzKSb2rbS9wDvdCwB4QnNRoMsC+UwhZx+3iy0FO77RWaj4GMrFjaw7SksjSJjnXZpyB/0IBtWU30Z/J1i6GQxfqK40sX/b/N230Wbdc11PZhlyzJRH0+EwlVgqJBJr1lHwbod+abjUqoh507IbDUtlUhmDeNRAaKtcBigifIr18=");
            if (r != FSDK.FSDKE_OK) {
                System.out.println("Not Working");
            }
        } catch (java.lang.UnsatisfiedLinkError e) {
            System.out.println(e.toString());
            System.exit(1);
        }
        
        FSDK.Initialize();

        toolkit = Toolkit.getDefaultToolkit();
        screenBounds = new Rectangle(toolkit.getScreenSize());

        
        //generateCsvFile(sFileName);

        try {
            writer = new FileWriter(sFileName);
            writer.append("FrameNumber");
            writer.append(',');
            writer.append("FeatureNumber");
            writer.append(',');
            writer.append("FeatureXAxis");
            writer.append(',');
            writer.append("FeatureYAxis");
            writer.append('\n');
            writer.close();

            TimerService service = new TimerService();
            service.setPeriod(Duration.seconds(1));
            service.start();
        } catch (Exception e) {

        }

//
        String workingDir = System.getProperty("user.dir");
        File f = new File(workingDir, "Chanaka Trim 1.mp4");
//
        Media m = new Media(f.toURI().toString());
        MediaPlayer mp = new MediaPlayer(m);
        MediaView mv = new MediaView(mp);

        StackPane root = new StackPane();
        root.getChildren().add(mv);

        stage.setScene(new Scene(root, 960, 540));
        stage.setFullScreen(true);
        stage.setTitle("Video Player");

        stage.show();
        mp.play();

    }

    public static void main(String[] args) {
        launch(args);

    }

    private class TimerService extends ScheduledService<Integer> {

        protected Task<Integer> createTask() {
            return new Task<Integer>() {
                protected Integer call() {
                    String fileName = i + ".png";
                    File someFile = new File(fileName);
                    dumpImageToFile(getDesktopScreenshot(), i);
                    HImage imageHandle = new HImage();
                    
                    try {
                        if (someFile.exists()) {
                            System.out.println("File " + someFile.getName() + " exists.");
                        }
                        if (FSDK.LoadImageFromFile(imageHandle, fileName) == FSDK.FSDKE_OK) {
                            Image awtImage[] = new Image[1];
                            if (FSDK.SaveImageToAWTImage(imageHandle, awtImage, FSDK.FSDK_IMAGEMODE.FSDK_IMAGE_COLOR_24BIT) == FSDK.FSDKE_OK) {
                                Image img = awtImage[0];
                                BufferedImage bimg = null;

                                FSDK.TFacePosition.ByReference facePosition = new FSDK.TFacePosition.ByReference();
                                if (FSDK.DetectFace(imageHandle, facePosition) != FSDK.FSDKE_OK) {
                                    System.out.println("No faces found");
                                } else {
                                    bimg = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);
                                    Graphics gr = bimg.getGraphics();
                                    gr.drawImage(img, 0, 0, null);
                                    gr.setColor(Color.green);

                                    int left = facePosition.xc - facePosition.w / 2;
                                    int top = facePosition.yc - facePosition.w / 2;
                                    gr.drawRect(left, top, facePosition.w, facePosition.w);

                                    FSDK_Features.ByReference facialFeatures = new FSDK_Features.ByReference();

                                    FSDK.DetectFacialFeaturesInRegion(imageHandle, (FSDK.TFacePosition) facePosition, facialFeatures);

                                    for (int j = 0; j < FSDK.FSDK_FACIAL_FEATURE_COUNT; ++j) {
                                        if (j < 2) {
                                            gr.setColor(Color.blue);
                                        } else if (j == 2) {
                                            gr.setColor(Color.green);
                                        }

                                        gr.drawOval(facialFeatures.features[j].x, facialFeatures.features[j].y, 3, 3);
                                    }
                                    TPoint[] features = facialFeatures.features;
                                    System.out.println("X: " + features[12].x + " Y: " + features[12].y);
                                    System.out.println("X: " + features[18].x + " Y: " + features[18].y);
                                    ArrayList<String[]> test = new ArrayList<String[]>();
                                    writer = new FileWriter(sFileName,true);
                                    
                                    for (int j = 0; j < FSDK.FSDK_FACIAL_FEATURE_COUNT; ++j) {
                                        writer.append(Integer.toString(i));
                                        writer.append(',');
                                        writer.append(Integer.toString(j));
                                        writer.append(',');
                                        writer.append(Integer.toString(features[j].x));
                                        writer.append(',');
                                        writer.append(Integer.toString(features[j].y));
                                        writer.append('\n');

                                    }
                                    writer.close();
                                    gr.dispose();
                                    
                                    //Save Image to file
                                    File outputFileName = new File("Output" + i + ".jpg");
                                    ImageIO.write(bimg, "png", outputFileName);

                                }
                            }
                            //writer.close();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    FSDK.FreeImage(imageHandle);

                    i++;
                    Integer something = 1;
                    return something;
                }
            };
        }
    }

    private String dumpImageToFile(BufferedImage image, int i) {
        try {
            String outputFilename = i + ".png";
            ImageIO.write(image, "png", new File(outputFilename));
            System.out.println(outputFilename);
            return outputFilename;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private static BufferedImage getDesktopScreenshot() {
        try {
            Robot robot = new Robot();
            Rectangle captureSize = new Rectangle(screenBounds);
            return robot.createScreenCapture(captureSize);
        } catch (AWTException e) {
            e.printStackTrace();
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

    }

}
