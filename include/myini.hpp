#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>

class IniFile {
public:
    IniFile(const std::string& filename) : filename_(filename) {}

    void create() {
        std::ofstream iniFile(filename_);
        if (iniFile.is_open()) {
            iniFile << "[Settings]\n";
            iniFile << "WindowWidth=800\n";
            iniFile << "WindowHeight=600\n";
            iniFile << "Fullscreen=false\n";

            iniFile << "[Config]\n";
            iniFile << "UserName=default_user\n";
            iniFile << "MaxRetries=5\n";

            iniFile.close();
        }
    }

    void load() {
        settings_.clear();
        std::ifstream iniFile(filename_);
        std::string line;
        while (std::getline(iniFile, line)) {
            if (line.empty() || line[0] == '[' || line[0] == ';') {
                continue;
            }
            std::istringstream lineStream(line);
            std::string key, value;
            if (std::getline(lineStream, key, '=') && std::getline(lineStream, value)) {
                settings_[key] = value;
            }
        }
    }

    void print() const {
        for (const auto& [key, value] : settings_) {
            std::cout << key << " = " << value << std::endl;
        }
    }

    std::string getValue(const std::string& key) const {
        auto it = settings_.find(key);
        if (it != settings_.end()) {
            return it->second;
        }
        return "";
    }

    void setValue(const std::string& key, const std::string& value) {
        settings_[key] = value;
    }

    void save() const {
        std::ofstream iniFile(filename_);
        if (iniFile.is_open()) {
            for (const auto& [key, value] : settings_) {
                iniFile << key << "=" << value << "\n";
            }
            iniFile.close();
        }
    }

private:
    std::string filename_;
    std::unordered_map<std::string, std::string> settings_;
};

//int main() {
//    const std::string iniFilename = "config.ini";
//    IniFile ini(iniFilename);
//
//    // 创建 ini 文件
//    ini.create();
//
//    // 加载 ini 文件
//    ini.load();
//
//    // 打印设置
//    ini.print();
//
//    // 获取单个值
//    std::cout << "WindowWidth: " << ini.getValue("WindowWidth") << std::endl;
//
//    // 修改并保存值
//    ini.setValue("WindowWidth", "1024");
//    ini.save();
//    std::cout << "WindowWidth: " << ini.getValue("WindowWidth") << std::endl;
//    return 0;
//}