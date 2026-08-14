package comm;

public class Packet {
    String src;
    String src_task_name;
    String entity_id;
    String dst;
    String txt;

    public String getSrc() { return src; }
    public void setSrc(String src) { this.src = src; }
    public String getSrc_task_name() { return src_task_name; }
    public void setSrc_task_name(String src_task_name) { this.src_task_name = src_task_name; }
    public String getEntity_id() { return entity_id; }
    public void setEntity_id(String entity_id) { this.entity_id = entity_id; }
    public String getDst() { return dst; }
    public void setDst(String dst) { this.dst = dst; }
    public String getTxt() { return txt; }
    public void setTxt(String txt) { this.txt = txt; }

    @Override
    public String toString() {
        return src + "?" + src_task_name + "?" + entity_id + "?" + dst + "?" + txt;
    }

    public static Packet toPacket(String s) {
        String[] info = s.split("\\?");
        Packet ret = new Packet();
        ret.src = info[0];
        ret.src_task_name = info[1];
        ret.entity_id = info[2];
        ret.dst = info[3];
        ret.txt = info[4];
        return ret;
    }
}
