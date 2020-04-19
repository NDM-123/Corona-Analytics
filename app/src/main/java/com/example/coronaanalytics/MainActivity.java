package com.example.coronaanalytics;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static com.example.coronaanalytics.FaceDataBase.faces;
import static com.example.coronaanalytics.FaceDataBase.recognize;

//import static com.example.coronaanalytics.FaceDataBase.faces;
//import static com.example.coronaanalytics.FaceDataBase.recognize;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    JavaCameraView javaCameraView;
    File cascFile;

    CascadeClassifier faceDetector;
    private Mat mRgba,mGrey;

    @Override
    protected void onCreate(Bundle savedInstancesState){
        super.onCreate(savedInstancesState);
        setContentView(R.layout.activity_main);
        readFaces();

        javaCameraView = (JavaCameraView)findViewById(R.id.javaCamView);

        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,baseCallback);
        }
        else{

                baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);


        }
        javaCameraView.setCvCameraViewListener(this);
    }

    private void readFaces() {
        try {
            String path =getFilesDir().getAbsolutePath();
            File f=new File(path, "profile.jpg");
            Bitmap b = BitmapFactory.decodeStream(new FileInputStream(f));

        Mat mat = new Mat();
        Bitmap bmp32 = b.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mat);
            faces.add(mat);
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
        }


    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mGrey = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGrey.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mGrey = inputFrame.gray();
        mRgba = inputFrame.rgba();

        //detect face
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mRgba,faceDetections);

        for(Rect rect: faceDetections.toArray()){
            Imgproc.rectangle(mRgba,new Point(rect.x,rect.y),
                    new Point(rect.x+rect.width,rect.y+rect.height),
                    new Scalar(255,0,0));
            Mat face =new Mat(null,rect);
           if(recognize(face,faces)){
             Imgproc.rectangle(mRgba,new Point(rect.x,rect.y),
                                new Point(rect.x+rect.width,rect.y+rect.height),
                                new Scalar(0,255,0));
             };//recognize change rectangle

        }
        return mRgba;
    }

    private BaseLoaderCallback baseCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status) {
                case LoaderCallbackInterface.SUCCESS: {
                    InputStream is = getResources().openRawResource((R.raw.haarcascade_frontalface_alt2));
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    cascFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
                    FileOutputStream fos = null;
                    try {
                        fos = new FileOutputStream(cascFile);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                    byte[] buffer = new byte[4096];
                    int bytesRead=0;

                    while (true) {
                        try {
                            if (!((bytesRead = is.read(buffer)) != -1)) break;
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        try {
                            fos.write(buffer, 0, bytesRead);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }

                    try {
                        is.close();
                        fos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    faceDetector = new CascadeClassifier((cascFile.getAbsolutePath()));

                    if (faceDetector.empty()) {
                        faceDetector = null;
                    } else {
                        cascadeDir.delete();

                        javaCameraView.enableView();
                    }
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }

            }
        }
    };

}
