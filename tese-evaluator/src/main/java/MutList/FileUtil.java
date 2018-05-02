package MutList;


import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

/**
 *
 * @author david
 */
public class FileUtil {

    public static InputStream getFile(InputStream input) {
        try {
            GZIPInputStream gis = new GZIPInputStream(input);
            return gis;
        }
        catch (IOException ex) {
            return input;
        }
    }

    public static class Filter implements FileFilter {

        private String[] extensions;

        public Filter(String[] extensions) {
            this.extensions = extensions;
        }

        @Override
        public boolean accept(File file) {
            for (String extension : extensions) {
                if (file.getName().toLowerCase().endsWith(extension)) {
                    return true;
                }
            }
            return false;
        }
    }
}
