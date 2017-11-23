package org.opencv.samples.tutorial1;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

public class Tutorial1Activity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "OCVSample::Activity";
    private static final String TESS_TRAINEDDATA_ASSETS_PATH = "tessdata/eng.traineddata";
    private static final boolean FACE_REC = false;
    private static final String FACE_CASCADE_ASSETS_PATH = "cascades/lbpcascade_frontalface_improved.xml";

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;

    private Mat displayedFrame = null;
    private Mat grayFrame = null;

    private String tesseractTrainedDataPath;
    private String faceCascadeDataPath;
    private TessBaseAPI mTess;
    List<String> words = new ArrayList<>();
    static final int WORD_SIZE = 40;


    CascadeClassifier face_cascade;

    private boolean stopReadingCameraInput;
    
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(Tutorial1Activity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public Tutorial1Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial1_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        tesseractTrainedDataPath = getFilesDir() + "/tesseract/";
        faceCascadeDataPath = getFilesDir().getPath() + "/";

        // make sure training data has been copied
        // first param needs the directory path under the assets file appended to it for the check.
        checkFile(new File(tesseractTrainedDataPath + "tessdata/"), tesseractTrainedDataPath, TESS_TRAINEDDATA_ASSETS_PATH);
        checkFile(new File(faceCascadeDataPath + "cascades/"), faceCascadeDataPath, FACE_CASCADE_ASSETS_PATH);

        String LANG = "eng";

        mTess = new TessBaseAPI();
        mTess.init(tesseractTrainedDataPath, LANG);
        stopReadingCameraInput = false;

//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

//        File cascadeFile = new File(faceCascadeDataPath + FACE_CASCADE_ASSETS_PATH);
//        face_cascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
//        if(!face_cascade.load(cascadeFile.getAbsolutePath())) {
//            //ERROR
//        }
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        if(FACE_REC) {
            detectAndDisplay(displayedFrame);
            return false;
        }

        if(!stopReadingCameraInput) {
            words = new ArrayList<>();

            Bitmap bmp = Bitmap.createBitmap(grayFrame.cols(), grayFrame.rows(), Bitmap.Config.ARGB_8888);
            Mat result = new Mat();
//            Imgproc.equalizeHist(grayFrame, grayFrame);
            Imgproc.adaptiveThreshold(grayFrame, result, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 61, 30);
            // display the processed, text readable image
            displayedFrame = result;
            Utils.matToBitmap(result, bmp);
            mTess.setImage(bmp);

            String screenText = mTess.getUTF8Text();
            //screenText = screenText.replaceAll("[^a-zA-Z0-9 +-=:]+", "");

            if(screenText.matches("[0-9]+ *[+-] *[0-9]+ *[=:]")) {
                int operatorIndex = screenText.indexOf("+");
                String operator = "+";
                if(operatorIndex < 0) {
                    operatorIndex = screenText.indexOf("-");
                    operator = "-";
                }

                int equalsIndex = screenText.indexOf("=");
                if (equalsIndex < 0) {
                    equalsIndex = screenText.indexOf(":");
                }
                String firstNum = screenText.substring(0, operatorIndex).trim();
                String secondNum = screenText.substring(operatorIndex + 1, equalsIndex).trim();
                String answer = "";
                if(operator == "+") {
                    answer = Integer.toString(Integer.valueOf(firstNum) + Integer.valueOf(secondNum));
                } else {
                    answer = Integer.toString(Integer.valueOf(firstNum) - Integer.valueOf(secondNum));
                }
                screenText += " " + answer;
            }

            for(int i = 0; i < screenText.length(); i += WORD_SIZE) {
                words.add(screenText.substring(i, Math.min(screenText.length(), i + WORD_SIZE)));
            }
        }

        stopReadingCameraInput = !stopReadingCameraInput;

		return false;
    }
    
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        if(!stopReadingCameraInput) {
            displayedFrame = inputFrame.rgba();
            grayFrame = inputFrame.gray();
        }

        for(int i = 0; i < words.size(); i++) {
            Imgproc.putText(
                    displayedFrame,
                    words.get(i),
                    new Point(50, 60 * (i + 1)),
                    Core.FONT_HERSHEY_SIMPLEX,
                    2,
                    new Scalar(0, 128, 255),
                    2
            );
        }
        return displayedFrame;
    }

    private void detectAndDisplay(Mat frame) {
        ArrayDeque<Rect> faces = new ArrayDeque<>();
        Mat gray = null;
        Imgproc.equalizeHist(grayFrame, gray);
    }

    private void copyTessTrainedDataFile(String filePath, String filePathInAssets) {
        try {
            AssetManager assetManager = getAssets();

            InputStream in = assetManager.open(filePathInAssets);
            OutputStream out = new FileOutputStream(filePath + filePathInAssets);

            //copy the file to the location specified by filepath
            byte[] buffer = new byte[1024];
            int readLength;
            while ((readLength = in.read(buffer)) != -1) {
                out.write(buffer, 0, readLength);
            }
            out.flush();
            out.close();
            in.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void checkFile(File dir, String filePath, String filePathInAssets) {
        if (!dir.exists() && dir.mkdirs()){
            copyTessTrainedDataFile(filePath, filePathInAssets);
        }

        if(dir.exists()) {

            File trainedData = new File(filePath + filePathInAssets);
            if (!trainedData.exists()) {
                copyTessTrainedDataFile(filePath, filePathInAssets);
            }
        }
    }
}
