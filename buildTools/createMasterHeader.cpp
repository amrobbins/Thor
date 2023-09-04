#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

bool readLine(FILE *fin, string &line) {
    line = "";
    if (feof(fin)) {
        return false;
    }
    for (char c = fgetc(fin); c != '\n' && c != EOF; c = fgetc(fin)) {
        line += c;
    }
    return true;
}

int main(int argc, char *argv[]) {
    assert(argc == 2);

    FILE *fin = fopen(argv[1], "r");
    assert(fin != NULL);

    FILE *fout = fopen("Thor.h", "w");
    assert(fout != NULL);

    vector<string> initialLines = {"#pragma once", ""};
    for (unsigned int i = 0; i < initialLines.size(); ++i)
        fputs((initialLines[i] + "\n").c_str(), fout);

    string line;
    while (readLine(fin, line)) {
        if (line.length() < 2)
            continue;
        if (line[0] != '.' || line[1] != '/')
            continue;
        if (line.find("/Thor.h") != string::npos)
            continue;

        // TEMP FIXME:
        if (line.find("GpuRTree") != string::npos)
            continue;

        string includeLine = string("#include \"") + line.substr(2).c_str() + string("\"\n");
        fputs(includeLine.c_str(), fout);
    }

    fclose(fout);
    fclose(fin);
}
