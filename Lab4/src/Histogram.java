import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Histogram {
    private static final int NUM_BINS = 5;

    public static class IntArrayWritable extends ArrayWritable {

        public IntArrayWritable() {
            super(IntWritable.class, null);
        }

        public IntArrayWritable(IntWritable[] values) {
            super(IntWritable.class, values);
        }

        @Override
        public IntWritable[] get() {
            Writable[] temp = super.get();
            if (temp != null) {
                int n = temp.length;
                IntWritable[] items = new IntWritable[n];
                for (int i = 0; i < n; i++) {
                    items[i] = (IntWritable)temp[i];
                }
                return items;
            } else {
                return null;
            }
        }

        @Override
        public String toString() {
            IntWritable[] values = get();
            StringBuilder sb = new StringBuilder();
            for (IntWritable v : values) {
                sb.append(v.get());
                sb.append(' ');
            }
            return sb.toString();
        }
    }

    public static class TokenizerMapper extends Mapper<Text, Text, Text, IntArrayWritable> {

        private final IntWritable[] def = new IntWritable[NUM_BINS];
        {
            for (int i = 0; i < def.length; ++i) {
                def[i] = new IntWritable(0);
            }
        }

        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            double v = Double.parseDouble(value.toString());
            double delta = 1./ NUM_BINS;
            int bin = 0;
            for (int i = 0; i < NUM_BINS; ++i) {
                if (delta*i <= v && v < delta*(i+1)) {
                    bin = i;
                    break;
                }
            }
            def[bin].set(1);
            context.write(key, new IntArrayWritable(def));
            def[bin].set(0);
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntArrayWritable, Text, IntArrayWritable> {

        @Override
        public void reduce(Text key, Iterable<IntArrayWritable> values, Context context)
                throws IOException, InterruptedException {
            IntWritable[] sum = new IntWritable[NUM_BINS];
            for (int i = 0; i < sum.length; ++i) {
                sum[i] = new IntWritable(0);
            }
            for (IntArrayWritable val : values) {
                IntWritable[] valArr = val.get();
                for (int i = 0; i < NUM_BINS; ++i) {
                    int res = sum[i].get()+valArr[i].get();
                    sum[i].set(res);
                }
            }
            context.write(key, new IntArrayWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "histogram");
        job.setJarByClass(Histogram.class);
        job.setJar("Histogram.jar");
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntArrayWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}