from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
    
    def get_clustering_model(self, image):
        """
        Reshapes the image to a 2D array and performs K-means clustering with 2 clusters.
        """
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extracts the dominant color of the player's jersey from the bounding box in the frame.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[:int(image.shape[0] / 2), :]  # Focus on the top half (jersey area)

        # Get clustering model for the top half of the player's bounding box
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to match the image shape
        clustered_image = labels.reshape(top_half_image.shape[:2])

        # Determine the cluster corresponding to the player's jersey
        corner_clusters = [
            clustered_image[0, 0], clustered_image[0, -1],
            clustered_image[-1, 0], clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get the RGB color of the player's cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assigns team colors based on clustering player colors from the first frame.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determines the team ID for a given player using their jersey color.
        """
        # Check if player ID is already assigned
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get player's color and predict their team
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Adjust from 0,1 to 1,2 for team IDs

        # Hardcoded exception for goalkeeper
        if player_id == 81:  # Assuming player ID 81 is the goalkeeper
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
    