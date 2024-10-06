/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Config.h"


namespace ORB_SLAM3
{

bool ConfigParser::ParseConfigFile(std::string &strConfigFile)
{
    return true;
}

// OptimizationParameters ReadGaussianptimParams() 
// {
//     std::filesystem::path json_path = "/home/ray/Desktop/ORB_SLAM3/src/Renderer/assets/optimization_params.json";
//     // Check if the file exists before trying to open it
//     if (!std::filesystem::exists(json_path)) {
//         throw std::runtime_error("Error: " + json_path.string() + " does not exist!");
//     }

//     std::ifstream file(json_path);
//     if (!file.is_open()) {
//         throw std::runtime_error("OptimizationParameter file could not be opened.");
//     }

//     std::stringstream buffer;
//     buffer << file.rdbuf();
//     std::string jsonString = buffer.str();
//     file.close(); // Explicitly close the file

//     // Parse the JSON string
//     nlohmann::json json = nlohmann::json::parse(jsonString);

//     OptimizationParameters params;
//     params.iterations = json["iterations"];
//     params.position_lr_init = json["position_lr_init"];
//     params.position_lr_final = json["position_lr_final"];
//     params.position_lr_delay_mult = json["position_lr_delay_mult"];
//     params.position_lr_max_steps = json["position_lr_max_steps"];
//     params.feature_lr = json["feature_lr"];
//     params.percent_dense = json["percent_dense"];
//     params.opacity_lr = json["opacity_lr"];
//     params.scaling_lr = json["scaling_lr"];
//     params.rotation_lr = json["rotation_lr"];
//     params.lambda_dssim = json["lambda_dssim"];
//     params.min_opacity = json["min_opacity"];
//     params.densification_interval = json["densification_interval"];
//     params.opacity_reset_interval = json["opacity_reset_interval"];
//     params.densify_from_iter = json["densify_from_iter"];
//     params.densify_until_iter = json["densify_until_iter"];
//     params.densify_grad_threshold = json["densify_grad_threshold"];
//     params.early_stopping = json["early_stopping"];
//     params.convergence_threshold = json["convergence_threshold"];
//     params.empty_gpu_cache = json["empty_gpu_cache"];

//     return params;
//         }

}
