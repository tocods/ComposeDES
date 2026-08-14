package comm;

import java.time.chrono.IsoEra;

public class Event {
    public String src_name;
    public String info;


    @Override
    public String toString() {
        return src_name + "_" + info;
    }

    public static Event toEvent(String s) {
        // NS-3 response format: "task_name\0\0...=sendend" (buffer padded with nulls)
        // Strip null bytes first, then parse
        String cleaned = s.replace("\0", "").trim();
        String taskPart = cleaned.split("=")[0];
        int colonIdx = taskPart.indexOf(':');
        String taskName = (colonIdx >= 0) ? taskPart.substring(0, colonIdx) : taskPart;

        Event e = new Event();
        e.src_name = taskName;
        e.info = "sendend";
        return e;
    }
}
